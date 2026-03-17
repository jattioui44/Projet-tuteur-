"""
Architecture PARALLÈLE - Swin + Capsules INDÉPENDANTS

NOUVELLE ARCHITECTURE:
                 ┌→ Swin Transformer → Swin Features
    Image ───────┤
                 └→ Capsules Network → Capsule Features
                           ↓
                        FUSION
                           ↓
                     Classification

DIFFÉRENCE CLÉE:
- AVANT: Capsules dépendent des features Swin
- APRÈS: Capsules traitent l'image DIRECTEMENT
- Les deux branches ont la MÊME IMPORTANCE
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
   
    # Hyperparamètres
    IMG_SIZE = 224
    BATCH_SIZE = 12  # Réduit car 2 branches parallèles
    NUM_EPOCHS = 100
    LEARNING_RATE = 3e-5
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Modèles
    SWIN_MODEL = 'swin_tiny_patch4_window7_224'
    
    # Capsules (traitement direct de l'image)
    USE_GRAYSCALE_FOR_CAPSULES = True  # True = grayscale (moins params), False = RGB
    NUM_PRIMARY_CAPS = 32
    PRIMARY_CAP_DIM = 8
    ROUTING_CAP_DIM = 16
    NUM_ROUTING_ITER = 3
    
    # Fusion
    FUSION_DIM = 256
    
    # Régularisation
    DROPOUT = 0.4
    DROP_PATH_RATE = 0.3
    WEIGHT_DECAY = 0.15
    LABEL_SMOOTHING = 0.1
    
    # Loss weights pour chaque branche
    SWIN_WEIGHT = 0.5
    CAPSULE_WEIGHT = 0.5
    
    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_parallel_model.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Squash Function ====================
def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)

# ==================== Primary Capsules ====================
class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size=9, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Convolution qui génère tous les capsules
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
    
    def forward(self, x):
        # x: [batch, in_channels, H, W]
        output = self.conv(x)  # [batch, num_caps*cap_dim, H', W']
        batch_size = output.size(0)
        
        # Reshape en capsules
        output = output.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        output = output.permute(0, 1, 3, 2).contiguous()  # [batch, num_caps, spatial, cap_dim]
        
        # Flatten spatial dimensions
        num_spatial = output.size(2)
        output = output.view(batch_size, self.num_capsules * num_spatial, self.capsule_dim)
        
        # Squash activation
        output = squash(output, dim=-1)
        
        return output

# ==================== Routing Capsules ====================
class RoutingCapsules(nn.Module):
    def __init__(self, num_in_caps, in_cap_dim, num_out_caps, out_cap_dim, num_iterations=3):
        super(RoutingCapsules, self).__init__()
        self.num_in_caps = num_in_caps
        self.in_cap_dim = in_cap_dim
        self.num_out_caps = num_out_caps
        self.out_cap_dim = out_cap_dim
        self.num_iterations = num_iterations
        
        # Transformation matrices
        self.W = nn.Parameter(torch.randn(num_in_caps, num_out_caps, out_cap_dim, in_cap_dim) * 0.01)
    
    def forward(self, u):
        # u: [batch, num_in_caps, in_cap_dim]
        batch_size = u.size(0)
        
        # u_hat = W @ u
        u_expanded = u.unsqueeze(2).unsqueeze(4)  # [batch, num_in, 1, in_dim, 1]
        W_expanded = self.W.unsqueeze(0)  # [1, num_in, num_out, out_dim, in_dim]
        u_hat = torch.matmul(W_expanded, u_expanded).squeeze(-1)  # [batch, num_in, num_out, out_dim]
        
        # Dynamic routing
        b = torch.zeros(batch_size, self.num_in_caps, self.num_out_caps, 1, device=u.device)
        
        for iteration in range(self.num_iterations):
            c = F.softmax(b, dim=2)  # [batch, num_in, num_out, 1]
            s = (c * u_hat).sum(dim=1)  # [batch, num_out, out_dim]
            v = squash(s, dim=-1)  # [batch, num_out, out_dim]
            
            if iteration < self.num_iterations - 1:
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1, keepdim=True)
                b = b + agreement
        
        return v

# ==================== Capsule Network (traitement direct image) ====================
class IndependentCapsuleNetwork(nn.Module):
    """
    Capsule Network qui traite l'image DIRECTEMENT
    (pas besoin des features Swin)
    """
    def __init__(self, num_classes=5, use_grayscale=True):
        super(IndependentCapsuleNetwork, self).__init__()
        
        self.use_grayscale = use_grayscale
        in_channels = 1 if use_grayscale else 3
        
        print(f"  📦 Capsules: Input channels = {in_channels} ({'Grayscale' if use_grayscale else 'RGB'})")
        
        # Convolution initiale pour extraction features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=9, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Primary Capsules
        self.primary_caps = PrimaryCapsules(
            in_channels=256,
            num_capsules=Config.NUM_PRIMARY_CAPS,
            capsule_dim=Config.PRIMARY_CAP_DIM,
            kernel_size=9,
            stride=2
        )
        
        # Calculer nombre de capsules après convolutions
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            dummy_caps = self.primary_caps(dummy)
            num_primary_caps_total = dummy_caps.size(1)
        
        print(f"  📦 Primary capsules total: {num_primary_caps_total}")
        
        # Routing Capsules
        self.routing_caps = RoutingCapsules(
            num_in_caps=num_primary_caps_total,
            in_cap_dim=Config.PRIMARY_CAP_DIM,
            num_out_caps=num_classes,
            out_cap_dim=Config.ROUTING_CAP_DIM,
            num_iterations=Config.NUM_ROUTING_ITER
        )
        
    def forward(self, x):
        # x: [batch, 3, 224, 224] ou [batch, 1, 224, 224]
        
        if self.use_grayscale and x.size(1) == 3:
            # Convertir RGB en grayscale
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Primary capsules
        x = self.primary_caps(x)  # [batch, num_caps, cap_dim]
        
        # Routing capsules
        x = self.routing_caps(x)  # [batch, num_classes, out_cap_dim]
        
        # Probabilités = longueur des vecteurs
        class_probs = torch.sqrt((x ** 2).sum(dim=-1))  # [batch, num_classes]
        
        return class_probs, x

# ==================== Architecture Parallèle ====================
class ParallelSwinCapsule(nn.Module):
    """
    Architecture parallèle:
    - Swin Transformer traite l'image
    - Capsules Network traite l'image (indépendamment)
    - Fusion des deux outputs
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(ParallelSwinCapsule, self).__init__()
        
        print("\n🔧 Construction architecture PARALLÈLE...")
        
        # ===== BRANCHE 1: Swin Transformer =====
        print("  📦 Branche 1: Swin Transformer")
        self.swin = timm.create_model(
            Config.SWIN_MODEL,
            pretrained=pretrained,
            num_classes=0,  # Pas de head
            drop_rate=Config.DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Dimension features Swin
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            swin_features = self.swin(dummy)
            self.swin_dim = swin_features.shape[1]
        
        print(f"    ✅ Swin features dim: {self.swin_dim}")
        
        # Classifier Swin
        self.swin_classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.swin_dim, num_classes)
        )
        
        # ===== BRANCHE 2: Capsule Network (INDÉPENDANT) =====
        print("  📦 Branche 2: Capsules Network (INDÉPENDANT)")
        self.capsule_net = IndependentCapsuleNetwork(
            num_classes=num_classes,
            use_grayscale=Config.USE_GRAYSCALE_FOR_CAPSULES
        )
        
        # ===== FUSION =====
        print("  🔀 Fusion des deux branches")
        
        # Dimension fusion = Swin features + Capsule vectors
        fusion_input_dim = self.swin_dim + (num_classes * Config.ROUTING_CAP_DIM)
        
        print(f"    Fusion input: {fusion_input_dim} ({self.swin_dim} + {num_classes * Config.ROUTING_CAP_DIM})")
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_DIM),
            nn.BatchNorm1d(Config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(Config.FUSION_DIM, Config.FUSION_DIM // 2),
            nn.BatchNorm1d(Config.FUSION_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(Config.FUSION_DIM // 2, num_classes)
        )
        
        print("✅ Architecture parallèle construite!\n")
        
    def forward(self, x):
        # x: [batch, 3, 224, 224]
        
        # Branche 1: Swin Transformer
        swin_features = self.swin(x)  # [batch, swin_dim]
        swin_logits = self.swin_classifier(swin_features)  # [batch, num_classes]
        
        # Branche 2: Capsules (traite image directement)
        capsule_probs, capsule_vectors = self.capsule_net(x)  # [batch, num_classes], [batch, num_classes, cap_dim]
        
        # Fusion
        capsule_flat = capsule_vectors.view(capsule_vectors.size(0), -1)  # [batch, num_classes * cap_dim]
        fusion_input = torch.cat([swin_features, capsule_flat], dim=1)  # [batch, fusion_input_dim]
        fusion_logits = self.fusion_layer(fusion_input)  # [batch, num_classes]
        
        return fusion_logits, swin_logits, capsule_probs

# ==================== Loss Hybride ====================
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    def forward(self, fusion_logits, swin_logits, capsule_logits, labels):
        fusion_loss = self.ce_loss(fusion_logits, labels)
        swin_loss = self.ce_loss(swin_logits, labels)
        capsule_loss = self.ce_loss(capsule_logits, labels)
        
        total_loss = fusion_loss + Config.SWIN_WEIGHT * swin_loss + Config.CAPSULE_WEIGHT * capsule_loss
        
        return total_loss, fusion_loss, swin_loss, capsule_loss

# ==================== Early Stopping ====================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, verbose=True):
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
    print("📁 Chargement données...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"Total: {len(df)}")
    
    print("\n📊 Distribution:")
    class_counts = df['diagnosis'].value_counts().sort_index()
    print(class_counts)
    
    # Détection format
    print("\n🔍 Détection format images...")
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
        print(f"\n⚠️  {len(missing)} images manquantes")
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

# ==================== Transformations ====================
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
        
        fusion_logits, swin_logits, capsule_logits = model(images)
        
        total_loss, fusion_loss, swin_loss, capsule_loss = criterion(
            fusion_logits, swin_logits, capsule_logits, labels
        )
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        _, predicted = fusion_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
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
            
            fusion_logits, swin_logits, capsule_logits = model(images)
            
            total_loss, _, _, _ = criterion(
                fusion_logits, swin_logits, capsule_logits, labels
            )
            
            running_loss += total_loss.item()
            _, predicted = fusion_logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), 100. * accuracy_score(all_labels, all_preds), all_preds, all_labels

# ==================== Main ====================
def main():
    print("\n" + "="*70)
    print("🚀 ARCHITECTURE PARALLÈLE - SWIN + CAPSULES INDÉPENDANTS")
    print("="*70)
    print("CONCEPT:")
    print("  📦 Branche 1: Swin Transformer (traite image RGB)")
    print("  📦 Branche 2: Capsules Network (traite image directement)")
    print(f"  📦 Capsules: {'Grayscale' if Config.USE_GRAYSCALE_FOR_CAPSULES else 'RGB'}")
    print("  🔀 Fusion: Égale importance (0.5 / 0.5)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ GPU non disponible!")
        return
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
    print("\n🤖 Construction modèle parallèle...")
    model = ParallelSwinCapsule(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Paramètres totaux: {total_params:,}")
    print(f"📊 Paramètres entraînables: {trainable_params:,}")
    
    # Loss et optimizer
    criterion = HybridLoss()
    
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
            print(f"✅ Meilleur modèle sauvegardé!")
        
        early_stopping(val_loss, epoch+1)
        if early_stopping.early_stop:
            break
    
    # Test
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n🧪 Évaluation finale\n")
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, Config.DEVICE)
    
    print(f"✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    print("\n📋 Rapport détaillé:")
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
    plt.title('Matrice de Confusion (Parallel Architecture)')
    plt.ylabel('Vraie')
    plt.xlabel('Prédite')
    plt.savefig('confusion_matrix_parallel.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Loss (Parallel)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Parallel)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_parallel.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print(f"🏆 Meilleur val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"🧪 Test accuracy: {test_acc:.2f}%")
    print("\n🎉 Entraînement terminé!")
    print(f"💾 Modèle sauvegardé: {Config.BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
