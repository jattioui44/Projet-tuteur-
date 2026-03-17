"""
Architecture PARALLÈLE - Swin + ConvNeXt (SIMPLE)

ARCHITECTURE:
                 ┌→ Swin Transformer → Swin Features [768]
    Image ───────┤
                 └→ ConvNeXt Tiny → ConvNeXt Features [768]
                           ↓
                        FUSION [1536]
                           ↓
                     Classification

SIMPLE & EFFICACE - Basé sur architecture qui fonctionnait
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image, ImageEnhance
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

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
    CONVNEXT_MODEL = 'convnext_tiny'
    
    # Fusion
    FUSION_DIM = 256
    
    # Régularisation AUGMENTÉE (fix overfitting)
    DROPOUT = 0.5           # ⬆️ 0.3 → 0.5 (fix Train 94% vs Val 82%)
    DROP_PATH_RATE = 0.2    # ⬆️ 0.1 → 0.2
    WEIGHT_DECAY = 0.15     # ⬆️ 0.1 → 0.15
    LABEL_SMOOTHING = 0.05  # ⬇️ 0.1 → 0.05 (moins smooth pour classes rares)
    
    # Loss weights pour chaque branche
    SWIN_WEIGHT = 0.5
    CONVNEXT_WEIGHT = 0.5
    
    # ==================== CLASS WEIGHTS ====================
    # Classe 3 (29 samples): 20.7% → besoin weight TRÈS fort
    # Classe 1 (56 samples): 33.9% → besoin weight fort
    CLASS_WEIGHTS = torch.tensor([0.3, 3.0, 1.0, 10.0, 2.5])  # Focus Classe 3!
    
    # Prétraitement
    USE_CONTRAST = True
    CONTRAST_FACTOR = 1.5
    
    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_parallel_swin_convnext_WEIGHTED.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Modèle Parallèle Simple ====================
class ParallelSwinConvNeXt(nn.Module):
    """
    Architecture parallèle simple:
    - Branche 1: Swin Transformer
    - Branche 2: ConvNeXt
    - Fusion: Concatenation simple
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(ParallelSwinConvNeXt, self).__init__()
        
        self.num_classes = num_classes
        
        print("\n🔨 Construction architecture parallèle...")
        
        # ==================== BRANCHE 1: SWIN TRANSFORMER ====================
        print("  📦 Branche 1: Swin Transformer Tiny")
        self.swin = timm.create_model(
            Config.SWIN_MODEL,
            pretrained=pretrained,
            num_classes=0,  # Sans classifier
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # ==================== BRANCHE 2: CONVNEXT ====================
        print("  📦 Branche 2: ConvNeXt Tiny")
        self.convnext = timm.create_model(
            Config.CONVNEXT_MODEL,
            pretrained=pretrained,
            num_classes=0,  # Sans classifier
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            swin_feat = self.swin(dummy)
            convnext_feat = self.convnext(dummy)
            
            swin_dim = swin_feat.shape[1]
            convnext_dim = convnext_feat.shape[1]
            fused_dim = swin_dim + convnext_dim
            
            print(f"\n  🔍 Feature Dimensions:")
            print(f"     Swin: {swin_dim}")
            print(f"     ConvNeXt: {convnext_dim}")
            print(f"     Fusionnées: {fused_dim}")
        
        # ==================== FUSION & CLASSIFIERS ====================
        # Classifier fusionné (sortie principale)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fused_dim, Config.FUSION_DIM),
            nn.BatchNorm1d(Config.FUSION_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.FUSION_DIM, num_classes)
        )
        
        # Classifiers auxiliaires (pour regularization)
        self.swin_classifier = nn.Linear(swin_dim, num_classes)
        self.convnext_classifier = nn.Linear(convnext_dim, num_classes)
        
        print(f"  🔀 Fusion: {fused_dim} → {Config.FUSION_DIM} → {num_classes}\n")
    
    def forward(self, x):
        # Branche 1: Swin
        swin_features = self.swin(x)
        swin_logits = self.swin_classifier(swin_features)
        
        # Branche 2: ConvNeXt
        convnext_features = self.convnext(x)
        convnext_logits = self.convnext_classifier(convnext_features)
        
        # Fusion
        fused_features = torch.cat([swin_features, convnext_features], dim=1)
        fusion_logits = self.fusion_classifier(fused_features)
        
        return fusion_logits, swin_logits, convnext_logits

# ==================== Loss Hybride avec Class Weights ====================
class HybridLoss(nn.Module):
    """
    Loss combinée avec class weights:
    - Fusion loss (principal) avec class weights
    - Swin loss (auxiliaire) avec class weights
    - ConvNeXt loss (auxiliaire) avec class weights
    """
    def __init__(self, class_weights=None):
        super(HybridLoss, self).__init__()
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights, 
                label_smoothing=Config.LABEL_SMOOTHING
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    def forward(self, fusion_logits, swin_logits, convnext_logits, labels):
        fusion_loss = self.ce_loss(fusion_logits, labels)
        swin_loss = self.ce_loss(swin_logits, labels)
        convnext_loss = self.ce_loss(convnext_logits, labels)
        
        total_loss = fusion_loss + Config.SWIN_WEIGHT * swin_loss + Config.CONVNEXT_WEIGHT * convnext_loss
        
        return total_loss, fusion_loss, swin_loss, convnext_loss

# ==================== Early Stopping ====================
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, min_delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.last_epoch = 0
        
    def __call__(self, val_loss, epoch):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'⏳ EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
        
        self.last_epoch = epoch

# ==================== Dataset ====================
class APTOSDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, use_contrast=False, contrast_factor=1.5):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_contrast = use_contrast
        self.contrast_factor = contrast_factor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.use_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.contrast_factor)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==================== Préparation données ====================
def prepare_data():
    print("\n📂 Chargement des données...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"  ✅ {len(df)} images chargées")
    
    # Full image paths
    image_paths = [os.path.join(Config.DATA_DIR, f"{id_code}.png") for id_code in df['id_code']]
    labels = df['diagnosis'].values
    
    # Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
    )
    
    print(f"  📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==================== Transforms ====================
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        fusion_logits, swin_logits, convnext_logits = model(images)
        
        total_loss, fusion_loss, swin_loss, convnext_loss = criterion(
            fusion_logits, swin_logits, convnext_logits, labels
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
        
        # Libérer mémoire
        del images, labels, fusion_logits, swin_logits, convnext_logits, total_loss
        torch.cuda.empty_cache()
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            fusion_logits, swin_logits, convnext_logits = model(images)
            
            total_loss, _, _, _ = criterion(
                fusion_logits, swin_logits, convnext_logits, labels
            )
            
            running_loss += total_loss.item()
            _, predicted = fusion_logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'acc': f'{100.*accuracy_score(all_labels, all_preds):.2f}%'
            })
            
            # Libérer mémoire
            del images, labels, fusion_logits, swin_logits, convnext_logits
            torch.cuda.empty_cache()
    
    return running_loss / len(dataloader), 100. * accuracy_score(all_labels, all_preds), all_preds, all_labels

# ==================== Main ====================
def main():
    print("\n" + "="*70)
    print("🚀 ARCHITECTURE PARALLÈLE - SWIN + CONVNEXT + CLASS WEIGHTS")
    print("="*70)
    print("ARCHITECTURE:")
    print("  📦 Branche 1: Swin Transformer Tiny 224")
    print("  📦 Branche 2: ConvNeXt Tiny 224")
    print("  🔀 Fusion: Concatenation simple [1536]")
    print("  🎯 Classifiers: Fusion + 2 auxiliaires")
    print(f"  ⚖️  Poids: Swin {Config.SWIN_WEIGHT} / ConvNeXt {Config.CONVNEXT_WEIGHT}")
    print(f"  ⚖️  Class Weights: {Config.CLASS_WEIGHTS}")
    print(f"  🛡️  Régularisation: Dropout {Config.DROPOUT}, DropPath {Config.DROP_PATH_RATE}")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ GPU non disponible!")
        return
    else:
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Données
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    train_dataset = APTOSDataset(X_train, y_train, train_transform, 
                                use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    val_dataset = APTOSDataset(X_val, y_val, val_test_transform,
                              use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    test_dataset = APTOSDataset(X_test, y_test, val_test_transform,
                               use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Modèle
    model = ParallelSwinConvNeXt(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Paramètres totaux: {total_params:,}")
    print(f"📊 Paramètres entraînables: {trainable_params:,}")
    
    # Loss et optimizer
    print(f"\n⚖️  Class Weights: {Config.CLASS_WEIGHTS}")
    criterion = HybridLoss(class_weights=Config.CLASS_WEIGHTS.to(Config.DEVICE))
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n" + "="*70)
    print("🎯 ENTRAÎNEMENT")
    print("="*70 + "\n")
    
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
        
        print(f"\n📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📈 Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, Config.BEST_MODEL_PATH)
            print(f"💾 Meilleur modèle sauvegardé! (Val Acc: {val_acc:.2f}%)")
        
        early_stopping(val_loss, epoch+1)
        if early_stopping.early_stop:
            print(f"\n⏹️  Early stopping à l'époque {epoch+1}")
            break
        
        print()
        
        # Libérer mémoire
        gc.collect()
        torch.cuda.empty_cache()
    
    # Test
    print("\n" + "="*70)
    print("🧪 ÉVALUATION FINALE")
    print("="*70)
    
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, Config.DEVICE)
    
    print(f"\n✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    
    if test_acc >= 85.0:
        print(f"\n🏆 EXCELLENT! {test_acc:.2f}% >= 85%!")
    elif test_acc >= 83.0:
        print(f"\n🎯 TRÈS BON! {test_acc:.2f}% >= 83%!")
    elif test_acc >= 80.0:
        print(f"\n✅ BON! {test_acc:.2f}% >= 80%!")
    
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
    plt.title(f'Parallel Swin+ConvNeXt WEIGHTED (Acc: {test_acc:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_parallel_WEIGHTED.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix: confusion_matrix_parallel_WEIGHTED.png")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss (Parallel Simple)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Parallel Simple)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_parallel_WEIGHTED.png', dpi=300, bbox_inches='tight')
    print(f"✅ Training curves: training_curves_parallel_WEIGHTED.png")
    
    print("\n" + "="*70)
    print(f"🏆 Meilleur val accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"🧪 Test accuracy: {test_acc:.2f}%")
    print("\n🎉 Entraînement terminé!")
    print(f"💾 Modèle sauvegardé: {Config.BEST_MODEL_PATH}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
