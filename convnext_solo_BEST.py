"""
ConvNeXt Tiny SEUL - Configuration OPTIMALE
Rétinopathie Diabétique APTOS2019

Meilleurs paramètres basés sur expérience:
- Dropout optimal (pas trop fort)
- Data augmentation forte
- Contrast enhancement
- Early stopping intelligent
- PAS de class weights (causent déséquilibre)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image, ImageEnhance
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# ==================== Configuration ====================
class Config:
    # Chemins
    DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_images"  # Dossier contenant les images 
    CSV_PATH = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train.csv"    
    
    # Modèle
    MODEL_NAME = 'convnext_tiny'
    # Reproductibilité
    SEED = 42
    NUM_WORKERS = 0  # Pour éviter les problèmes de multiprocessing sur Windows
    # Hyperparamètres OPTIMAUX
    IMG_SIZE = 224
    BATCH_SIZE = 64  # Plus grand que 12 pour stabilité
    NUM_EPOCHS = 100
    LEARNING_RATE = 3e-5
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Régularisation OPTIMALE
    DROPOUT = 0.3  # Optimal (pas 0.5 trop fort)
    DROP_PATH_RATE = 0.1
    WEIGHT_DECAY = 0.1
    LABEL_SMOOTHING = 0.0  # Désactivé (simplicité)
    
    # Prétraitement
    USE_CONTRAST = True
    CONTRAST_FACTOR = 1.5
    
    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_convnext_solo.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Modèle ConvNeXt ====================
class ConvNeXtClassifier(nn.Module):
    """ConvNeXt Tiny avec classifier simple et efficace"""
    def __init__(self, num_classes=5, dropout=0.3):
        super(ConvNeXtClassifier, self).__init__()
        
        # ConvNeXt backbone
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,  # Transfer learning
            num_classes=0,  # Pas de classifier
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
        
        # Classifier SIMPLE et EFFICACE
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ==================== Early Stopping ====================
class EarlyStopping:
    def __init__(self, patience=15, verbose=True, min_delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
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
        
        # Contrast enhancement
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
    
    # Distribution classes
    print("\n📊 Distribution classes:")
    for i in range(Config.NUM_CLASSES):
        count = (df['diagnosis'] == i).sum()
        print(f"  Classe {i}: {count} images ({100*count/len(df):.1f}%)")
    
    image_paths = [os.path.join(Config.DATA_DIR, f"{id_code}.png") for id_code in df['id_code']]
    labels = df['diagnosis'].values
    
    # Split stratifié
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
    )
    
    print(f"\n  📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==================== Transforms OPTIMALES ====================
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    # Augmentation FORTE
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
        
        del images, labels, outputs, loss
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'acc': f'{100.*accuracy_score(all_labels, all_preds):.2f}%'
            })
            
            del images, labels, outputs
            torch.cuda.empty_cache()
    
    return running_loss / len(dataloader), 100. * accuracy_score(all_labels, all_preds), all_preds, all_labels
# ==================== Reproductibilité ====================
def set_seed(seed=42):
    """Fixe tous les seeds pour reproductibilité"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Si multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
# ==================== Main ====================
def main():
    # Fixer les seeds pour reproductibilité
    set_seed(42)
    print("\n" + "="*80)
    print("🚀 CONVNEXT TINY SOLO - CONFIGURATION OPTIMALE")
    print("="*80)
    print(f"MODÈLE: {Config.MODEL_NAME}")
    print(f"IMAGE SIZE: {Config.IMG_SIZE}")
    print(f"BATCH SIZE: {Config.BATCH_SIZE}")
    print(f"LEARNING RATE: {Config.LEARNING_RATE}")
    print(f"DROPOUT: {Config.DROPOUT}")
    print(f"DROP PATH: {Config.DROP_PATH_RATE}")
    print(f"WEIGHT DECAY: {Config.WEIGHT_DECAY}")
    print(f"CONTRAST: {Config.CONTRAST_FACTOR if Config.USE_CONTRAST else 'Disabled'}")
    print("="*80)
    
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
                        shuffle=True, num_workers=0,  # ← 0 au lieu de Config.NUM_WORKERS
                        pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Modèle
    print("\n🔨 Construction modèle ConvNeXt...")
    model = ConvNeXtClassifier(num_classes=Config.NUM_CLASSES, dropout=Config.DROPOUT)
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  📊 Paramètres totaux: {total_params:,}")
    print(f"  📊 Paramètres entraînables: {trainable_params:,}")
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf') 
    best_epoch = 0
    
    print("\n" + "="*80)
    print("🎯 ENTRAÎNEMENT")
    print("="*80 + "\n")
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Époque [{epoch+1}/{Config.NUM_EPOCHS}]")
        print("-"*80)
        
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
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, Config.BEST_MODEL_PATH)
            print(f"💾 Meilleur modèle sauvegardé! (Val Loss: {val_loss:.4f})")
        
        early_stopping(val_loss, epoch+1)
        if early_stopping.early_stop:
            print(f"\n⏹️  Early stopping à l'époque {epoch+1}")
            break
        
        print()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Test
    print("\n" + "="*80)
    print("🧪 ÉVALUATION FINALE")
    print("="*80)
    
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, Config.DEVICE)
    
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    
    # Kappa
    kappa = cohen_kappa_score(test_labels, test_preds, weights='quadratic')
    print(f"📊 Quadratic Kappa: {kappa:.4f}")
    
    if test_acc >= 85.0:
        print(f"\n🏆 EXCELLENT! {test_acc:.2f}% >= 85%!")
    elif test_acc >= 82.0:
        print(f"\n🎯 TRÈS BON! {test_acc:.2f}% >= 82%!")
    elif test_acc >= 80.0:
        print(f"\n✅ BON! {test_acc:.2f}% >= 80%!")
    
    print("\n📋 Rapport détaillé:")
    print(classification_report(test_labels, test_preds, 
                               target_names=[f'Classe {i}' for i in range(Config.NUM_CLASSES)],
                               zero_division=0))
    
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
    plt.title(f'ConvNeXt Solo (Acc: {test_acc:.1f}%, Kappa: {kappa:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_convnext_solo.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix: confusion_matrix_convnext_solo.png")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss (ConvNeXt Solo)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (ConvNeXt Solo)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_convnext_solo.png', dpi=300, bbox_inches='tight')
    print(f"✅ Training curves: training_curves_convnext_solo.png")
    
    print("\n" + "="*80)
    print(f"🏆 Meilleur val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"🧪 Test accuracy: {test_acc:.2f}%")
    print(f"📊 Quadratic Kappa: {kappa:.4f}")
    print("\n🎉 Entraînement terminé!")
    print(f"💾 Modèle sauvegardé: {Config.BEST_MODEL_PATH}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
