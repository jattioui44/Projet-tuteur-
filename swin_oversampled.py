"""
Solution OVERSAMPLING + SWIN seul (sans Capsules)

STRATÉGIE:
1. Over-sampling intelligent (SMOTE ou duplication)
2. Swin Transformer seul (plus simple, plus efficace)
3. Class weights modérés
4. Augmentation forte sur classes minoritaires
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ==================== Configuration ====================
class Config:
    # Chemins
    DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_images"  # Dossier contenant les images 
    CSV_PATH = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train.csv" 
    
    # Hyperparamètres
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Augmenté car plus simple
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5  # Légèrement augmenté
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Modèle - SWIN SEUL (sans capsules)
    MODEL_NAME = 'swin_small_patch4_window7_224'  # Small au lieu de Tiny
    
    # Oversampling
    USE_OVERSAMPLING = True
    OVERSAMPLE_STRATEGY = 'weighted_sampler'  # 'weighted_sampler' ou 'duplicate'
    TARGET_RATIO = 0.5  # Ratio cible pour classes minoritaires
    
    # Early Stopping
    PATIENCE = 12
    MIN_DELTA = 0.001
    
    # Régularisation
    DROPOUT = 0.3
    DROP_PATH_RATE = 0.2
    WEIGHT_DECAY = 0.1
    LABEL_SMOOTHING = 0.1
    
    # Class Weights - MODÉRÉS (pas trop agressifs)
    USE_CLASS_WEIGHTS = True
    WEIGHT_SCALE = 0.5  # Réduire l'intensité des weights
    
    # Augmentation spécifique par classe
    STRONG_AUGMENT_MINORITY = True  # Plus forte pour 1,3,4
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_swin_oversampled.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Modèle Simple (Swin seul) ====================
class SwinClassifier(nn.Module):
    """Swin Transformer simple (sans capsules)"""
    def __init__(self, num_classes=5, pretrained=True):
        super(SwinClassifier, self).__init__()
        
        self.swin = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,  # Pas de head
            drop_rate=Config.DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Obtenir dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.swin(dummy)
            feature_dim = features.shape[1]
        
        print(f"✅ Swin feature dim: {feature_dim}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.swin(x)
        logits = self.classifier(features)
        return logits

# ==================== Loss ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==================== Early Stopping ====================
class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.001, verbose=True):
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
    def __init__(self, image_paths, labels, transform=None, strong_transform=None, 
                 use_strong_for_minority=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.strong_transform = strong_transform
        self.use_strong_for_minority = use_strong_for_minority
        
        # Classes minoritaires: 1, 3, 4
        self.minority_classes = [1, 3, 4]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Augmentation forte pour classes minoritaires
        if (self.use_strong_for_minority and 
            label in self.minority_classes and 
            self.strong_transform is not None):
            image = self.strong_transform(image)
        elif self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== Préparation Données ====================
def prepare_data():
    print("📁 Chargement...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"Total: {len(df)}")
    
    print("\n📊 Distribution originale:")
    class_counts = df['diagnosis'].value_counts().sort_index()
    print(class_counts)
    
    # Calcul class weights MODÉRÉS
    if Config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df['diagnosis']),
            y=df['diagnosis']
        )
        # Réduire l'intensité des weights
        class_weights = 1 + (class_weights - 1) * Config.WEIGHT_SCALE
        class_weights = torch.FloatTensor(class_weights).to(Config.DEVICE)
        
        print("\n⚖️  Class Weights MODÉRÉS:")
        for i, w in enumerate(class_weights):
            print(f"   Classe {i}: {w:.3f} (count: {class_counts[i]})")
    else:
        class_weights = None
    
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
    
    # Oversampling avec WeightedRandomSampler
    train_sampler = None
    if Config.USE_OVERSAMPLING and Config.OVERSAMPLE_STRATEGY == 'weighted_sampler':
        print("\n🔄 Création Weighted Sampler...")
        
        class_counts_train = Counter(y_train)
        total_train = len(y_train)
        
        # Calculer poids pour chaque échantillon
        class_weights_sample = {}
        for cls in range(Config.NUM_CLASSES):
            count = class_counts_train[cls]
            # Poids inversement proportionnel à la fréquence
            class_weights_sample[cls] = total_train / (Config.NUM_CLASSES * count)
        
        # Appliquer aux échantillons
        sample_weights = [class_weights_sample[label] for label in y_train]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print("✅ Weighted Sampler créé!")
        print("   Distribution effective (approximative):")
        for cls in range(Config.NUM_CLASSES):
            effective_samples = int(class_weights_sample[cls] * class_counts_train[cls])
            print(f"   Classe {cls}: ~{effective_samples} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights, train_sampler

# ==================== Transformations ====================

# Augmentation standard
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Augmentation FORTE pour classes minoritaires
strong_train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),  # Plus fort
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
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
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
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), 100. * accuracy_score(all_labels, all_preds), all_preds, all_labels

# ==================== Main ====================
def main():
    print("\n" + "="*70)
    print("🚀 SWIN SEUL + OVERSAMPLING")
    print("="*70)
    print("STRATÉGIE:")
    print("✅ Swin Transformer seul (sans capsules)")
    print(f"✅ Oversampling: {Config.OVERSAMPLE_STRATEGY}")
    print(f"✅ Class Weights modérés (scale={Config.WEIGHT_SCALE})")
    print(f"✅ Augmentation forte pour minoritaires")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ GPU non disponible!")
        return
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Données
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights, train_sampler = prepare_data()
    
    # Datasets
    train_dataset = APTOSDataset(
        X_train, y_train, 
        transform=train_transform,
        strong_transform=strong_train_transform,
        use_strong_for_minority=Config.STRONG_AUGMENT_MINORITY
    )
    val_dataset = APTOSDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = APTOSDataset(X_test, y_test, transform=val_test_transform)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,  # Utilise sampler si oversampling
        shuffle=(train_sampler is None),  # Shuffle seulement si pas de sampler
        num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # Modèle SIMPLE
    print("\n🤖 Construction modèle (Swin seul)...")
    model = SwinClassifier(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Paramètres: {total_params:,}")
    
    # Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
    
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
    
    # Confusion matrix par classe
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
    plt.title('Matrice de Confusion (Oversampled)')
    plt.ylabel('Vraie')
    plt.xlabel('Prédite')
    plt.savefig('confusion_matrix_oversampled.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Loss (Oversampled)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Oversampled)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_oversampled.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print(f"🏆 Best val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"🧪 Test acc: {test_acc:.2f}%")
    print("\n🎉 Terminé!")

if __name__ == "__main__":
    main()
