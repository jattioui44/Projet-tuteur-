"""
Architecture Hybride: Swin Transformer + Capsule Network
Pour classification APTOS 2019 (Rétinopathie Diabétique)

INNOVATION:
• Swin Transformer: Extraction de features hiérarchiques
• Capsule Network: Meilleure représentation des relations spatiales
• Fusion: Concaténation des deux pour classification finale
• Taille optimisée: 192x192 (économie mémoire)

AVANTAGES:
• Capsules capturent mieux les relations partie-tout
• Meilleure robustesse aux variations d'orientation
• Fusion améliore la discrimination des classes
• Adapté pour GPU avec mémoire limitée (8-10 GB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ==================== Configuration ====================
class Config:
    # Chemins
    DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_images"  # Dossier contenant les images 
    CSV_PATH = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train.csv"     # Fichier CSV avec labels
    
    
    # Hyperparamètres - OPTIMISÉS POUR ARCHITECTURE HYBRIDE
    IMG_SIZE = 224           # ← 224x224 (modèle standard de timm)
    BATCH_SIZE = 12          # ← Ajusté pour l'architecture hybride
    NUM_EPOCHS = 40
    LEARNING_RATE = 3e-5     # ← LR réduit pour stabilité
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Modèle Swin Transformer
    MODEL_NAME = 'swin_tiny_patch4_window7_224'  # ← Modèle standard timm
    # Note: C'est le seul Swin-Tiny disponible dans timm
    
    # Capsule Network
    NUM_PRIMARY_CAPS = 32    # Nombre de capsules primaires
    PRIMARY_CAP_DIM = 8      # Dimension de chaque capsule primaire
    NUM_ROUTING_CAPS = 5     # = NUM_CLASSES (une capsule par classe)
    ROUTING_CAP_DIM = 16     # Dimension de chaque capsule de routage
    NUM_ROUTING_ITER = 3     # Itérations du dynamic routing
    
    # Architecture hybride
    SWIN_FEATURE_DIM = 768   # Dimension features Swin-Tiny
    FUSION_DIM = 256         # Dimension après fusion
    
    # Early Stopping
    PATIENCE = 7
    MIN_DELTA = 0.001
    
    # Régularisation
    DROPOUT = 0.3            # ← Augmenté pour architecture plus complexe
    DROP_PATH_RATE = 0.2
    WEIGHT_DECAY = 0.1
    LABEL_SMOOTHING = 0.1
    
    # Loss weights (pour combiner les pertes)
    CAPSULE_LOSS_WEIGHT = 0.7
    SWIN_LOSS_WEIGHT = 0.3
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_swin_capsule_model.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Capsule Network Components ====================

def squash(tensor, dim=-1):
    """
    Squashing function pour normaliser les capsules
    Préserve la direction mais normalise la magnitude
    """
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)

class PrimaryCapsules(nn.Module):
    """
    Capsules primaires: Première couche de capsules
    Convertit les features CNN en capsules
    """
    def __init__(self, in_channels, num_capsules, cap_dim, kernel_size=9, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.cap_dim = cap_dim
        
        # Convolution pour créer les capsules
        self.conv = nn.Conv2d(
            in_channels, 
            num_capsules * cap_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        
    def forward(self, x):
        # x: [batch, in_channels, H, W]
        output = self.conv(x)  # [batch, num_caps*cap_dim, H', W']
        
        batch_size = output.shape[0]
        
        # Reshape en capsules
        output = output.view(
            batch_size, 
            self.num_capsules, 
            self.cap_dim, 
            -1
        )  # [batch, num_caps, cap_dim, H'*W']
        
        # Permute pour avoir: [batch, num_caps*H'*W', cap_dim]
        output = output.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch_size, -1, self.cap_dim)
        
        # Squash
        output = squash(output, dim=-1)
        
        return output

class RoutingCapsules(nn.Module):
    """
    Capsules de routage: Capsules de niveau supérieur avec dynamic routing
    Implémente le routing by agreement
    """
    def __init__(self, num_in_caps, in_cap_dim, num_out_caps, out_cap_dim, num_iterations=3):
        super(RoutingCapsules, self).__init__()
        self.num_in_caps = num_in_caps
        self.num_out_caps = num_out_caps
        self.num_iterations = num_iterations
        
        # Matrice de transformation pour chaque paire (in_cap, out_cap)
        self.W = nn.Parameter(
            torch.randn(1, num_in_caps, num_out_caps, out_cap_dim, in_cap_dim)
        )
        
    def forward(self, x):
        # x: [batch, num_in_caps, in_cap_dim]
        batch_size = x.shape[0]
        
        # Expand pour multiplication
        x_expanded = x.unsqueeze(2).unsqueeze(4)
        # [batch, num_in_caps, 1, in_cap_dim, 1]
        
        W_expanded = self.W.expand(batch_size, -1, -1, -1, -1)
        # [batch, num_in_caps, num_out_caps, out_cap_dim, in_cap_dim]
        
        # Prédictions: u_hat
        u_hat = torch.matmul(W_expanded, x_expanded).squeeze(-1)
        # [batch, num_in_caps, num_out_caps, out_cap_dim]
        
        # Dynamic routing
        b = torch.zeros(batch_size, self.num_in_caps, self.num_out_caps, 1).to(x.device)
        
        for iteration in range(self.num_iterations):
            # Softmax sur les routing coefficients
            c = F.softmax(b, dim=2)  # [batch, num_in_caps, num_out_caps, 1]
            
            # Weighted sum
            s = (c * u_hat).sum(dim=1)  # [batch, num_out_caps, out_cap_dim]
            
            # Squash
            v = squash(s, dim=-1)
            
            # Update routing coefficients (sauf à la dernière itération)
            if iteration < self.num_iterations - 1:
                # Agreement: dot product entre v et u_hat
                v_expanded = v.unsqueeze(1)  # [batch, 1, num_out_caps, out_cap_dim]
                agreement = (u_hat * v_expanded).sum(dim=-1, keepdim=True)
                # [batch, num_in_caps, num_out_caps, 1]
                b = b + agreement
        
        return v  # [batch, num_out_caps, out_cap_dim]

class CapsuleNetwork(nn.Module):
    """
    Capsule Network complet pour classification
    """
    def __init__(self, input_dim, num_classes, num_primary_caps=32, primary_cap_dim=8, 
                 routing_cap_dim=16, num_routing_iter=3):
        super(CapsuleNetwork, self).__init__()
        
        self.num_classes = num_classes
        
        # Projection des features Swin vers format spatial
        # Pour créer une "pseudo-image" pour les primary capsules
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 256 * 6 * 6),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Primary Capsules
        self.primary_caps = PrimaryCapsules(
            in_channels=256,
            num_capsules=num_primary_caps,
            cap_dim=primary_cap_dim,
            kernel_size=3,
            stride=1
        )
        
        # Calculer le nombre de capsules primaires après convolution
        # Avec kernel=3, stride=1 sur 6x6: output = 4x4 = 16 positions
        num_primary_caps_total = num_primary_caps * 16
        
        # Routing Capsules
        self.routing_caps = RoutingCapsules(
            num_in_caps=num_primary_caps_total,
            in_cap_dim=primary_cap_dim,
            num_out_caps=num_classes,
            out_cap_dim=routing_cap_dim,
            num_iterations=num_routing_iter
        )
        
    def forward(self, x):
        # x: [batch, input_dim]
        
        # Projeter vers format spatial
        x = self.feature_projection(x)  # [batch, 256*6*6]
        x = x.view(-1, 256, 6, 6)  # [batch, 256, 6, 6]
        
        # Primary capsules
        x = self.primary_caps(x)  # [batch, num_primary_caps_total, cap_dim]
        
        # Routing capsules
        x = self.routing_caps(x)  # [batch, num_classes, routing_cap_dim]
        
        # Longueur des capsules = probabilités des classes
        class_probs = torch.sqrt((x ** 2).sum(dim=-1))  # [batch, num_classes]
        
        return class_probs, x

# ==================== Architecture Hybride ====================

class SwinCapsuleHybrid(nn.Module):
    """
    Architecture hybride combinant:
    1. Swin Transformer (extraction de features)
    2. Capsule Network (représentation spatiale)
    3. Fusion (concaténation + classification)
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(SwinCapsuleHybrid, self).__init__()
        
        self.num_classes = num_classes
        
        # 1. Swin Transformer Backbone
        self.swin = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,  # Pas de classification head (on prend les features)
            drop_rate=Config.DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Obtenir la dimension des features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            swin_features = self.swin(dummy_input)
            swin_feature_dim = swin_features.shape[1]
        
        print(f"Swin feature dimension: {swin_feature_dim}")
        
        # 2. Capsule Network
        self.capsule_net = CapsuleNetwork(
            input_dim=swin_feature_dim,
            num_classes=num_classes,
            num_primary_caps=Config.NUM_PRIMARY_CAPS,
            primary_cap_dim=Config.PRIMARY_CAP_DIM,
            routing_cap_dim=Config.ROUTING_CAP_DIM,
            num_routing_iter=Config.NUM_ROUTING_ITER
        )
        
        # 3. Classification head pour Swin seul (voie parallèle)
        self.swin_classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(swin_feature_dim, num_classes)
        )
        
        # 4. Fusion layer (combine Swin + Capsules)
        # Dimension: swin_features + capsule_routing_vectors
        fusion_input_dim = swin_feature_dim + (num_classes * Config.ROUTING_CAP_DIM)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(Config.FUSION_DIM, Config.FUSION_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM // 2, num_classes)
        )
        
    def forward(self, x):
        # 1. Extraire features avec Swin
        swin_features = self.swin(x)  # [batch, swin_feature_dim]
        
        # 2. Prédiction Swin seule (voie parallèle)
        swin_logits = self.swin_classifier(swin_features)  # [batch, num_classes]
        
        # 3. Capsule Network
        capsule_probs, capsule_vectors = self.capsule_net(swin_features)
        # capsule_probs: [batch, num_classes]
        # capsule_vectors: [batch, num_classes, routing_cap_dim]
        
        # 4. Fusion
        # Flatten les capsule vectors
        capsule_flat = capsule_vectors.view(capsule_vectors.size(0), -1)
        # [batch, num_classes * routing_cap_dim]
        
        # Concaténer Swin features + Capsule vectors
        fusion_input = torch.cat([swin_features, capsule_flat], dim=1)
        
        # Prédiction finale fusionnée
        fusion_logits = self.fusion_layer(fusion_input)  # [batch, num_classes]
        
        return fusion_logits, swin_logits, capsule_probs

# ==================== Loss Functions ====================

class HybridLoss(nn.Module):
    """
    Loss combinée pour l'architecture hybride:
    - CrossEntropy pour fusion (principale)
    - CrossEntropy pour Swin (auxiliaire)
    - Margin loss pour Capsules (auxiliaire)
    """
    def __init__(self, label_smoothing=0.1):
        super(HybridLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def margin_loss(self, capsule_probs, labels, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """
        Margin loss pour Capsule Network
        """
        batch_size = capsule_probs.size(0)
        
        # One-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=Config.NUM_CLASSES).float()
        
        # Margin loss
        loss_positive = labels_one_hot * F.relu(m_plus - capsule_probs) ** 2
        loss_negative = lambda_ * (1 - labels_one_hot) * F.relu(capsule_probs - m_minus) ** 2
        
        loss = (loss_positive + loss_negative).sum(dim=1).mean()
        
        return loss
    
    def forward(self, fusion_logits, swin_logits, capsule_probs, labels):
        # Loss principale: Fusion
        loss_fusion = self.ce_loss(fusion_logits, labels)
        
        # Loss auxiliaire: Swin
        loss_swin = self.ce_loss(swin_logits, labels)
        
        # Loss auxiliaire: Capsules
        loss_capsule = self.margin_loss(capsule_probs, labels)
        
        # Combiner les losses
        total_loss = (
            loss_fusion + 
            Config.SWIN_LOSS_WEIGHT * loss_swin + 
            Config.CAPSULE_LOSS_WEIGHT * loss_capsule
        )
        
        return total_loss, loss_fusion, loss_swin, loss_capsule

# ==================== Early Stopping ====================
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'\n🛑 Early Stopping! Meilleur: époque {self.best_epoch}')
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0

# ==================== Dataset ====================
class APTOSDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== Préparation Données ====================
def prepare_data():
    print("📁 Chargement des données...")
    
    df = pd.read_csv(Config.CSV_PATH)
    print(f"Nombre total d'images: {len(df)}")
    
    print("\n📊 Distribution des classes:")
    print(df['diagnosis'].value_counts().sort_index())
    
    # Détection format fichiers
    print("\n🔍 Détection du format des noms de fichiers...")
    image_paths = []
    missing_files = []
    
    for img_id in df['id_code']:
        possible_paths = [
            os.path.join(Config.DATA_DIR, f"{img_id}_m3.png"),
            os.path.join(Config.DATA_DIR, f"{img_id}.png"),
            os.path.join(Config.DATA_DIR, f"{img_id}.jpg"),
            os.path.join(Config.DATA_DIR, f"{img_id}_m3.jpg"),
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                image_paths.append(path)
                found = True
                break
        
        if not found:
            missing_files.append(img_id)
    
    if image_paths:
        print(f"✅ Format détecté: {os.path.basename(image_paths[0])}")
        print(f"✅ Images trouvées: {len(image_paths)}/{len(df)}")
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} fichiers non trouvés!")
        response = input("❓ Continuer? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    labels = df['diagnosis'].values
    
    if len(image_paths) < len(labels):
        labels = np.array([labels[i] for i, img_id in enumerate(df['id_code']) 
                          if img_id not in missing_files])
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.6, random_state=42, stratify=y_temp
    )
    
    print(f"\n✅ Train: {len(X_train)} ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"✅ Validation: {len(X_val)} ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"✅ Test: {len(X_test)} ({len(X_test)/len(image_paths)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==================== Transformations ====================
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== Training ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_fusion_loss = 0.0
    running_swin_loss = 0.0
    running_capsule_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        fusion_logits, swin_logits, capsule_probs = model(images)
        
        # Loss
        loss, loss_fusion, loss_swin, loss_capsule = criterion(
            fusion_logits, swin_logits, capsule_probs, labels
        )
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_fusion_loss += loss_fusion.item()
        running_swin_loss += loss_swin.item()
        running_capsule_loss += loss_capsule.item()
        
        _, predicted = fusion_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    losses = {
        'total': epoch_loss,
        'fusion': running_fusion_loss / len(dataloader),
        'swin': running_swin_loss / len(dataloader),
        'capsule': running_capsule_loss / len(dataloader)
    }
    
    return epoch_loss, epoch_acc, losses

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            fusion_logits, swin_logits, capsule_probs = model(images)
            
            loss, _, _, _ = criterion(fusion_logits, swin_logits, capsule_probs, labels)
            
            running_loss += loss.item()
            _, predicted = fusion_logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# ==================== Main ====================
def main():
    print("\n" + "="*80)
    print("🚀 ARCHITECTURE HYBRIDE: SWIN TRANSFORMER + CAPSULE NETWORK")
    print("="*80)
    
    # GPU Check
    print("\n🖥️  Vérification GPU:")
    if not torch.cuda.is_available():
        print("❌ GPU non disponible!")
        response = input("Continuer sur CPU? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Données
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    train_dataset = APTOSDataset(X_train, y_train, transform=train_transform)
    val_dataset = APTOSDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = APTOSDataset(X_test, y_test, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # Modèle hybride
    print("\n" + "="*80)
    print("🤖 Construction du modèle hybride")
    print("="*80)
    print(f"Taille image: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"Swin Transformer: {Config.MODEL_NAME}")
    print(f"Capsule Network:")
    print(f"  • Primary capsules: {Config.NUM_PRIMARY_CAPS} x {Config.PRIMARY_CAP_DIM}D")
    print(f"  • Routing capsules: {Config.NUM_ROUTING_CAPS} x {Config.ROUTING_CAP_DIM}D")
    print(f"  • Routing iterations: {Config.NUM_ROUTING_ITER}")
    
    model = SwinCapsuleHybrid(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Paramètres totaux: {total_params:,}")
    print(f"📊 Paramètres entraînables: {trainable_params:,}")
    
    # Loss, Optimizer, Scheduler
    criterion = HybridLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True)
    
    # Historique
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Entraînement
    print("\n" + "="*80)
    print("🎯 Début de l'entraînement")
    print("="*80)
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nÉpoque [{epoch+1}/{Config.NUM_EPOCHS}]")
        print("-"*80)
        
        train_loss, train_acc, train_losses = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n📈 Résultats:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"     └─ Fusion: {train_losses['fusion']:.4f}")
        print(f"     └─ Swin: {train_losses['swin']:.4f}")
        print(f"     └─ Capsule: {train_losses['capsule']:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, Config.BEST_MODEL_PATH)
            print(f"   ✅ Meilleur modèle sauvegardé!")
        
        early_stopping(val_loss, epoch+1)
        if early_stopping.early_stop:
            break
    
    # Test final
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "="*80)
    print("🧪 Évaluation finale sur TEST")
    print("="*80)
    
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, Config.DEVICE
    )
    
    print(f"\n✅ Loss test: {test_loss:.4f}")
    print(f"✅ Accuracy test: {test_acc:.2f}%")
    print("\n📋 Rapport de Classification:")
    print(classification_report(test_labels, test_preds, 
                               target_names=[f'Classe {i}' for i in range(Config.NUM_CLASSES)]))
    
    # Visualisations
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion (Swin+Capsule Hybrid)')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.savefig('confusion_matrix_hybrid.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Loss (Hybrid Architecture)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Hybrid Architecture)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_hybrid.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*80)
    print("📊 RÉSUMÉ FINAL")
    print("="*80)
    print(f"🏆 Meilleure val loss: {best_val_loss:.4f} (Époque {best_epoch})")
    print(f"🧪 Test accuracy: {test_acc:.2f}%")
    print(f"📏 Architecture: Swin-Tiny + Capsule Network")
    print(f"📏 Taille image: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print("\n🎉 Terminé!")

if __name__ == "__main__":
    main()
