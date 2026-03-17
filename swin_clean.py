"""
VERSION CLEAN: Swin seul SANS class weights

PHILOSOPHIE: Laisser le modèle apprendre naturellement
- Pas de class weights agressifs
- Pas de focal loss fort
- Juste un bon modèle avec bonne régularisation
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
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Modèle - Swin Small (plus capacité)
    MODEL_NAME = 'swin_small_patch4_window7_224'
    
    # Régularisation modérée
    DROPOUT = 0.3
    DROP_PATH_RATE = 0.2
    WEIGHT_DECAY = 0.05  # Réduit
    LABEL_SMOOTHING = 0.1
    
    # Early Stopping
    PATIENCE = 12
    MIN_DELTA = 0.001
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_swin_clean.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Modèle Simple ====================
class SwinClassifier(nn.Module):
    """Swin Transformer simple et efficace"""
    def __init__(self, num_classes=5, pretrained=True):
        super(SwinClassifier, self).__init__()
        
        self.swin = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=Config.DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.swin(dummy)
            feature_dim = features.shape[1]
        
        print(f"✅ Swin feature dim: {feature_dim}")
        
        # Classification head simple
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.swin(x)
        logits = self.classifier(features)
        return logits

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
    print("🚀 SWIN CLEAN (Sans class weights)")
    print("="*70)
    print("PHILOSOPHIE:")
    print("✅ Swin Small (plus de capacité)")
    print("✅ SANS class weights (apprentissage naturel)")
    print("✅ CrossEntropy + Label Smoothing")
    print("✅ Régularisation modérée")
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
    print("\n🤖 Construction modèle...")
    model = SwinClassifier(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Paramètres: {total_params:,}")
    
    # Loss SIMPLE - CrossEntropy avec label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
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
    plt.title('Matrice de Confusion (Clean)')
    plt.ylabel('Vraie')
    plt.xlabel('Prédite')
    plt.savefig('confusion_matrix_clean.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Loss (Clean)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Clean)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_clean.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print(f"🏆 Best val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"🧪 Test acc: {test_acc:.2f}%")
    print("\n🎉 Terminé!")

if __name__ == "__main__":
    main()
