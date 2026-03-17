"""
VERSION CAPSULES SANS CLASS WEIGHTS

OBJECTIF: Tester l'hypothèse de l'utilisateur proprement
- GARDER: Swin + Capsules (même architecture que balanced)
- ENLEVER: Class Weights SEULEMENT
- Tout le reste IDENTIQUE à balanced

Ainsi on peut comparer:
- Balanced (Capsules + Weights) → 83%
- Cette version (Capsules + NO Weights) → ?%
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

# ==================== Configuration ====================
class Config:
    # Chemins
    DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_images"  # Dossier contenant les images 
    CSV_PATH = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train.csv" 
    
    # Hyperparamètres - IDENTIQUES à balanced
    IMG_SIZE = 224
    BATCH_SIZE = 12
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-5
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Modèle - IDENTIQUE à balanced
    MODEL_NAME = 'swin_tiny_patch4_window7_224'
    
    # Capsule - IDENTIQUES à balanced
    NUM_CAPSULES = 5
    CAPSULE_DIM = 16
    NUM_ROUTING_ITERATIONS = 3
    PRIMARY_CAPS_DIM = 8
    NUM_PRIMARY_CAPSULES = 32
    
    # Loss weights - IDENTIQUES à balanced
    SWIN_LOSS_WEIGHT = 0.5
    CAPSULE_LOSS_WEIGHT = 0.5
    
    # Régularisation - IDENTIQUES à balanced
    DROPOUT = 0.4
    DROP_PATH_RATE = 0.3
    WEIGHT_DECAY = 0.15
    LABEL_SMOOTHING = 0.1
    
    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # CHANGEMENT: Class Weights DÉSACTIVÉS
    USE_CLASS_WEIGHTS = False  # ← SEULE DIFFÉRENCE!
    USE_FOCAL_LOSS = False      # ← Pas de focal loss non plus
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_capsules_no_weights.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Primary Capsules ====================
class PrimaryCapsules(nn.Module):
    """IDENTIQUE à balanced"""
    def __init__(self, in_channels, num_capsules=32, capsule_dim=8):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim, 
                             kernel_size=1, stride=1)
        
    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, self.capsule_dim)
        return self.squash(x)

# ==================== Routing Capsules ====================
class RoutingCapsules(nn.Module):
    """IDENTIQUE à balanced"""
    def __init__(self, in_capsules, in_dim, num_capsules, capsule_dim, 
                 num_iterations=3):
        super(RoutingCapsules, self).__init__()
        self.in_capsules = in_capsules
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        
        self.W = nn.Parameter(torch.randn(1, in_capsules, num_capsules, 
                                         capsule_dim, in_dim))
        
    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.unsqueeze(2).unsqueeze(2)
        W_batch = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W_batch, x.unsqueeze(-1)).squeeze(-1)
        
        b = torch.zeros(batch_size, self.in_capsules, self.num_capsules, 1,
                       device=x.device)
        
        for iteration in range(self.num_iterations):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=False)
            v = self.squash(s)
            
            if iteration < self.num_iterations - 1:
                agreement = torch.matmul(u_hat, v.unsqueeze(-1))
                b = b + agreement
        
        return v

# ==================== Hybrid Model ====================
class SwinCapsuleHybrid(nn.Module):
    """IDENTIQUE à balanced - Architecture complète"""
    def __init__(self, num_classes=5, pretrained=True):
        super(SwinCapsuleHybrid, self).__init__()
        
        self.swin = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=Config.DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Déterminer la dimension de sortie
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.swin.forward_features(dummy)  # [1, 49, 768]
            self.swin_dim = features.shape[2]  # 768
        
        print(f"✅ Swin feature dim: {self.swin_dim}")
        
        self.swin_classifier = nn.Sequential(
            nn.LayerNorm(self.swin_dim),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.swin_dim, num_classes)
        )
        
        self.conv_proj = nn.Conv2d(self.swin_dim, 256, kernel_size=1)
        
        self.primary_capsules = PrimaryCapsules(
            in_channels=256,
            num_capsules=Config.NUM_PRIMARY_CAPSULES,
            capsule_dim=Config.PRIMARY_CAPS_DIM
        )
        
        primary_caps_total = Config.NUM_PRIMARY_CAPSULES * 49
        
        self.routing_capsules = RoutingCapsules(
            in_capsules=primary_caps_total,
            in_dim=Config.PRIMARY_CAPS_DIM,
            num_capsules=Config.NUM_CAPSULES,
            capsule_dim=Config.CAPSULE_DIM,
            num_iterations=Config.NUM_ROUTING_ITERATIONS
        )
        
        fusion_input_dim = self.swin_dim + (Config.NUM_CAPSULES * Config.CAPSULE_DIM)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extraire features de plusieurs niveaux du Swin
        # On va utiliser forward_features au lieu de forward
        x_swin = self.swin.forward_features(x)  # Donne [batch, H*W, dim]
        
        # Pooling pour classification
        swin_features = x_swin.mean(dim=1)  # [batch, 768]
        swin_logits = self.swin_classifier(swin_features)
        
        # Pour capsules, on reshape en feature map
        # x_swin a shape [batch, 49, 768] (7*7 patches)
        H = W = 7
        feature_map = x_swin.transpose(1, 2).contiguous()  # [batch, 768, 49]
        feature_map = feature_map.view(batch_size, self.swin_dim, H, W)  # [batch, 768, 7, 7]
        feature_map = self.conv_proj(feature_map)  # [batch, 256, 7, 7]
        
        primary_caps = self.primary_capsules(feature_map)
        digit_caps = self.routing_capsules(primary_caps)
        
        capsule_features = digit_caps.view(batch_size, -1)
        
        fused_features = torch.cat([swin_features, capsule_features], dim=1)
        fusion_logits = self.fusion_layer(fused_features)
        
        capsule_probs = torch.norm(digit_caps, dim=-1)
        
        return fusion_logits, swin_logits, capsule_probs

# ==================== Loss SANS Class Weights ====================
class HybridLoss(nn.Module):
    """Loss SANS class weights - SEULE DIFFÉRENCE"""
    def __init__(self):
        super(HybridLoss, self).__init__()
        # CrossEntropy SIMPLE - pas de weights!
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        
    def margin_loss(self, capsule_probs, labels):
        """Margin loss SANS class weights"""
        batch_size = capsule_probs.size(0)
        labels_one_hot = F.one_hot(labels, num_classes=Config.NUM_CLASSES).float()
        
        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5
        
        loss_positive = labels_one_hot * F.relu(m_plus - capsule_probs) ** 2
        loss_negative = lambda_val * (1.0 - labels_one_hot) * F.relu(capsule_probs - m_minus) ** 2
        
        loss = (loss_positive + loss_negative).sum(dim=1)
        return loss.mean()
    
    def forward(self, fusion_logits, swin_logits, capsule_probs, labels):
        # Loss fusion
        fusion_loss = self.ce_loss(fusion_logits, labels)
        
        # Loss Swin
        swin_loss = self.ce_loss(swin_logits, labels)
        
        # Loss Capsule (margin loss)
        capsule_loss = self.margin_loss(capsule_probs, labels)
        
        # Combinaison IDENTIQUE à balanced
        total_loss = (fusion_loss + 
                     Config.SWIN_LOSS_WEIGHT * swin_loss + 
                     Config.CAPSULE_LOSS_WEIGHT * capsule_loss)
        
        return total_loss, fusion_loss, swin_loss, capsule_loss

# ==================== Early Stopping ====================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
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
                print(f'   EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'\n🛑 Early Stop! Best: epoch {self.best_epoch}')
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

# ==================== Préparation ====================
def prepare_data():
    print("📁 Chargement...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"Total: {len(df)}")
    
    print("\n📊 Distribution:")
    class_counts = df['diagnosis'].value_counts().sort_index()
    print(class_counts)
    
    # Détection format
    print("\n🔍 Détection format...")
    image_paths = []
    missing = []
    
    for img_id in df['id_code']:
        possible = [
            os.path.join(Config.DATA_DIR, f"{img_id}_m3.png"),
            os.path.join(Config.DATA_DIR, f"{img_id}.png"),
            os.path.join(Config.DATA_DIR, f"{img_id}.jpg"),
        ]
        
        found = False
        for path in possible:
            if os.path.exists(path):
                image_paths.append(path)
                found = True
                break
        
        if not found:
            missing.append(img_id)
    
    if image_paths:
        print(f"✅ Format: {os.path.basename(image_paths[0])}")
        print(f"✅ Trouvées: {len(image_paths)}/{len(df)}")
    
    if missing:
        print(f"\n⚠️  {len(missing)} manquants")
        response = input("Continuer? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    labels = df['diagnosis'].values
    if len(image_paths) < len(labels):
        labels = np.array([labels[i] for i, img_id in enumerate(df['id_code']) 
                          if img_id not in missing])
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.6, random_state=42, stratify=y_temp
    )
    
    print(f"\n✅ Train: {len(X_train)}")
    print(f"✅ Val: {len(X_val)}")
    print(f"✅ Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==================== Transformations - IDENTIQUES à balanced ====================
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        fusion_logits, swin_logits, capsule_probs = model(images)
        
        total_loss, fusion_loss, swin_loss, capsule_loss = criterion(
            fusion_logits, swin_logits, capsule_probs, labels
        )
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        _, predicted = fusion_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(dataloader), 100. * correct / total

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
            
            total_loss, _, _, _ = criterion(
                fusion_logits, swin_logits, capsule_probs, labels
            )
            
            running_loss += total_loss.item()
            _, predicted = fusion_logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), 100. * accuracy_score(all_labels, all_preds), all_preds, all_labels

# ==================== Main ====================
def main():
    print("\n" + "="*70)
    print("🚀 SWIN + CAPSULES SANS CLASS WEIGHTS")
    print("="*70)
    print("TEST PROPRE de l'hypothèse:")
    print("  Architecture: IDENTIQUE à balanced (Swin + Capsules)")
    print("  SEULE DIFFÉRENCE: SANS class weights")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ GPU non disponible!")
        return
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Données
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    train_dataset = APTOSDataset(X_train, y_train, train_transform)
    val_dataset = APTOSDataset(X_val, y_val, val_test_transform)
    test_dataset = APTOSDataset(X_test, y_test, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # Modèle
    print("\n🤖 Construction modèle (Swin + Capsules)...")
    model = SwinCapsuleHybrid(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Paramètres: {total_params:,}")
    
    # Loss SANS class weights
    criterion = HybridLoss()
    print("✅ Loss: CrossEntropy + Margin Loss (SANS class weights)")
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("\n🎯 Entraînement\n")
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Époque [{epoch+1}/{Config.NUM_EPOCHS}]")
        print("-"*70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, Config.DEVICE)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n📈 Train: {train_loss:.4f} | {train_acc:.2f}%")
        print(f"📈 Val:   {val_loss:.4f} | {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, Config.BEST_MODEL_PATH)
            print(f"✅ Meilleur modèle!")
        
        early_stopping(val_loss, epoch+1)
        if early_stopping.early_stop:
            break
    
    # Test
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n🧪 Test final\n")
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, Config.DEVICE)
    
    print(f"✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Acc: {test_acc:.2f}%")
    print("\n📋 Rapport:")
    print(classification_report(test_labels, test_preds, 
                               target_names=[f'Classe {i}' for i in range(Config.NUM_CLASSES)]))
    
    # Accuracy par classe
    cm = confusion_matrix(test_labels, test_preds)
    print("\n📊 Accuracy par classe:")
    for i in range(Config.NUM_CLASSES):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_acc = 100. * class_correct / class_total if class_total > 0 else 0
        print(f"   Classe {i}: {class_correct}/{class_total} = {class_acc:.1f}%")
    
    # Visualisations
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion (Capsules sans Weights)')
    plt.ylabel('Vraie')
    plt.xlabel('Prédite')
    plt.savefig('confusion_matrix_capsules_no_weights.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Loss (Capsules sans Weights)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Capsules sans Weights)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_capsules_no_weights.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("📊 COMPARAISON:")
    print(f"Balanced (avec weights): 83%")
    print(f"Cette version (sans weights): {test_acc:.2f}%")
    print("="*70)
    print("\n🎉 Terminé!")

if __name__ == "__main__":
    main()