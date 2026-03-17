"""
Architecture Hybride SWIN + EFFICIENTNET pour APTOS 2019

CONCEPT:
- Swin Transformer: Extraction features globales (attention)
- EfficientNet: Extraction features locales (convolutions)
- Fusion intelligente des deux backbones
- Classification finale

AVANTAGES:
- Complémentarité CNN (local) + Transformer (global)
- Double extraction de features
- Plus robuste et précis
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
    BATCH_SIZE = 8  # Plus petit car 2 backbones = plus de mémoire
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-5
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Modèles
    SWIN_MODEL = 'swin_small_patch4_window7_224'  # Swin Small
    EFFICIENTNET_MODEL = 'efficientnet_b3'  # EfficientNet-B3
    
    # Fusion
    FUSION_METHOD = 'concat'  # 'concat', 'add', 'attention'
    DROPOUT = 0.4
    
    # Loss weights pour chaque backbone
    SWIN_WEIGHT = 0.4
    EFFICIENT_WEIGHT = 0.4
    FUSION_WEIGHT = 0.6  # Le plus important
    
    # Régularisation
    WEIGHT_DECAY = 0.1
    LABEL_SMOOTHING = 0.1
    DROP_PATH_RATE = 0.2
    
    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_swin_efficient_hybrid.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Attention Fusion Module ====================
class AttentionFusion(nn.Module):
    """
    Fusion par attention: le modèle apprend quel backbone écouter
    """
    def __init__(self, dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x1, x2):
        # x1: features Swin [batch, dim]
        # x2: features EfficientNet [batch, dim]
        
        # Concatener
        concat = torch.cat([x1, x2], dim=1)
        
        # Calculer attention weights
        weights = self.attention(concat)  # [batch, 2]
        
        # Appliquer attention
        w1 = weights[:, 0:1]  # [batch, 1]
        w2 = weights[:, 1:2]  # [batch, 1]
        
        fused = w1 * x1 + w2 * x2
        
        return fused

# ==================== Modèle Hybride ====================
class SwinEfficientNetHybrid(nn.Module):
    """
    Modèle hybride Swin Transformer + EfficientNet
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(SwinEfficientNetHybrid, self).__init__()
        
        print("🔧 Construction modèle hybride...")
        
        # ===== BACKBONE 1: Swin Transformer =====
        print("  📦 Chargement Swin Transformer...")
        self.swin = timm.create_model(
            Config.SWIN_MODEL,
            pretrained=pretrained,
            num_classes=0,  # Pas de classification head
            drop_rate=Config.DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Dimension features Swin
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            swin_features = self.swin(dummy)
            self.swin_dim = swin_features.shape[1]
        
        print(f"    ✅ Swin dim: {self.swin_dim}")
        
        # ===== BACKBONE 2: EfficientNet =====
        print("  📦 Chargement EfficientNet...")
        self.efficientnet = timm.create_model(
            Config.EFFICIENTNET_MODEL,
            pretrained=pretrained,
            num_classes=0,  # Pas de classification head
            drop_rate=Config.DROPOUT
        )
        
        # Dimension features EfficientNet
        with torch.no_grad():
            efficient_features = self.efficientnet(dummy)
            self.efficient_dim = efficient_features.shape[1]
        
        print(f"    ✅ EfficientNet dim: {self.efficient_dim}")
        
        # ===== Adaptation des dimensions =====
        # Projeter les deux à la même dimension pour fusion
        self.target_dim = 512
        
        self.swin_proj = nn.Sequential(
            nn.Linear(self.swin_dim, self.target_dim),
            nn.LayerNorm(self.target_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        self.efficient_proj = nn.Sequential(
            nn.Linear(self.efficient_dim, self.target_dim),
            nn.LayerNorm(self.target_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        # ===== Classification Heads (pour chaque backbone) =====
        self.swin_classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.target_dim, num_classes)
        )
        
        self.efficient_classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.target_dim, num_classes)
        )
        
        # ===== Fusion Module =====
        print(f"  🔀 Fusion method: {Config.FUSION_METHOD}")
        
        if Config.FUSION_METHOD == 'concat':
            fusion_input_dim = self.target_dim * 2
        elif Config.FUSION_METHOD == 'add':
            fusion_input_dim = self.target_dim
        elif Config.FUSION_METHOD == 'attention':
            self.attention_fusion = AttentionFusion(self.target_dim)
            fusion_input_dim = self.target_dim
        
        # Classification finale (fusion)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        print("✅ Modèle hybride construit!")
        
    def forward(self, x):
        # Extraire features des 2 backbones
        swin_features = self.swin(x)  # [batch, swin_dim]
        efficient_features = self.efficientnet(x)  # [batch, efficient_dim]
        
        # Projeter à même dimension
        swin_proj = self.swin_proj(swin_features)  # [batch, 512]
        efficient_proj = self.efficient_proj(efficient_features)  # [batch, 512]
        
        # Classification individuelle (pour loss auxiliaire)
        swin_logits = self.swin_classifier(swin_proj)
        efficient_logits = self.efficient_classifier(efficient_proj)
        
        # Fusion
        if Config.FUSION_METHOD == 'concat':
            fused = torch.cat([swin_proj, efficient_proj], dim=1)
        elif Config.FUSION_METHOD == 'add':
            fused = swin_proj + efficient_proj
        elif Config.FUSION_METHOD == 'attention':
            fused = self.attention_fusion(swin_proj, efficient_proj)
        
        # Classification finale
        fusion_logits = self.fusion_classifier(fused)
        
        return fusion_logits, swin_logits, efficient_logits

# ==================== Loss Hybride ====================
class HybridLoss(nn.Module):
    """
    Loss qui combine les 3 sorties:
    - Swin branch
    - EfficientNet branch
    - Fusion branch (le plus important)
    """
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    def forward(self, fusion_logits, swin_logits, efficient_logits, labels):
        # Loss sur chaque branche
        fusion_loss = self.ce_loss(fusion_logits, labels)
        swin_loss = self.ce_loss(swin_logits, labels)
        efficient_loss = self.ce_loss(efficient_logits, labels)
        
        # Loss totale pondérée
        total_loss = (Config.FUSION_WEIGHT * fusion_loss + 
                     Config.SWIN_WEIGHT * swin_loss + 
                     Config.EFFICIENT_WEIGHT * efficient_loss)
        
        return total_loss, fusion_loss, swin_loss, efficient_loss

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
        
        fusion_logits, swin_logits, efficient_logits = model(images)
        
        total_loss, fusion_loss, swin_loss, efficient_loss = criterion(
            fusion_logits, swin_logits, efficient_logits, labels
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
            
            fusion_logits, swin_logits, efficient_logits = model(images)
            
            total_loss, _, _, _ = criterion(
                fusion_logits, swin_logits, efficient_logits, labels
            )
            
            running_loss += total_loss.item()
            _, predicted = fusion_logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), 100. * accuracy_score(all_labels, all_preds), all_preds, all_labels

# ==================== Main ====================
def main():
    print("\n" + "="*70)
    print("🚀 SWIN + EFFICIENTNET HYBRID")
    print("="*70)
    print("ARCHITECTURE:")
    print(f"  📦 Swin: {Config.SWIN_MODEL}")
    print(f"  📦 EfficientNet: {Config.EFFICIENTNET_MODEL}")
    print(f"  🔀 Fusion: {Config.FUSION_METHOD}")
    print(f"  ⚖️  Weights: Swin={Config.SWIN_WEIGHT}, Eff={Config.EFFICIENT_WEIGHT}, Fusion={Config.FUSION_WEIGHT}")
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
    print("\n🤖 Construction modèle hybride...")
    model = SwinEfficientNetHybrid(num_classes=Config.NUM_CLASSES, pretrained=True)
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
    plt.title('Matrice de Confusion (Swin+EfficientNet)')
    plt.ylabel('Vraie')
    plt.xlabel('Prédite')
    plt.savefig('confusion_matrix_hybrid.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Loss (Hybrid)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Hybrid)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_hybrid.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print(f"🏆 Meilleur val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"🧪 Test accuracy: {test_acc:.2f}%")
    print("\n🎉 Entraînement terminé!")
    print(f"💾 Modèle sauvegardé: {Config.BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
