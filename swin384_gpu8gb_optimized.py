"""
SWIN 384 OPTIMISÉ - VERSION GPU 8GB
────────────────────────────────────

Basé sur meilleure version testée (#5)
+ Optimisations pour atteindre 89%
+ Adapté GPU 8GB (VRAM ~7-7.5GB)

Architecture:
• Swin Base 384
• Ordinal Regression
• Prétraitement Contraste
• TTA (Test-Time Augmentation)
• Gradient Accumulation (simule batch 16)

Objectif: 88-89% accuracy
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# ==================== Configuration GPU 8GB ====================
class Config:
    # Chemins - À MODIFIER
    DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_images"  # Dossier contenant les images 
    CSV_PATH = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train.csv" 
    
    # ==================== GPU 8GB OPTIMISÉ ====================
    IMG_SIZE = 384  # Garder 384 pour qualité
    BATCH_SIZE = 6  # Réduit pour GPU 8GB
    ACCUMULATION_STEPS = 3  # Simule batch 18 (6×3)
    
    # ==================== TRAINING ====================
    NUM_EPOCHS = 100
    LEARNING_RATE = 1.5e-5
    NUM_CLASSES = 5
    NUM_WORKERS = 4  # Réduit pour GPU 8GB
    
    # ==================== MODÈLE ====================
    SWIN_MODEL = 'swin_base_patch4_window12_384'
    HIDDEN_DIM = 512
    
    # ==================== RÉGULARISATION ====================
    DROPOUT = 0.6
    DROP_PATH_RATE = 0.5
    WEIGHT_DECAY = 0.25
    LABEL_SMOOTHING = 0.2
    
    # ==================== ORDINAL REGRESSION ====================
    USE_ORDINAL = True
    
    # ==================== CLASS WEIGHTS (OPTIMISÉS!) ====================
    CLASS_WEIGHTS = torch.tensor([0.3, 2.0, 1.2, 4.5, 2.0])
    
    # ==================== PRÉTRAITEMENT ====================
    USE_CONTRAST = True
    CONTRAST_FACTOR = 1.5
    
    # ==================== MIXUP ====================
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    
    # ==================== TEST-TIME AUGMENTATION ====================
    USE_TTA = True
    TTA_TRANSFORMS = 4
    
    # ==================== MIXED PRECISION (OBLIGATOIRE GPU 8GB!) ====================
    USE_MIXED_PRECISION = True
    
    # ==================== EARLY STOPPING ====================
    PATIENCE = 20
    MIN_DELTA = 0.0005
    
    # ==================== SAUVEGARDE ====================
    BEST_MODEL_PATH = "./best_swin384_gpu8gb.pth"
    
    # ==================== DEVICE ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def print_config():
        print("\n" + "="*70)
        print("📋 CONFIGURATION SWIN 384 - GPU 8GB OPTIMISÉ")
        print("="*70)
        print(f"  📸 Image size: {Config.IMG_SIZE}×{Config.IMG_SIZE}")
        print(f"  📦 Batch size: {Config.BATCH_SIZE} (effective: {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS})")
        print(f"  🔄 Gradient Accumulation: {Config.ACCUMULATION_STEPS} steps")
        print(f"  🧠 Swin Model: {Config.SWIN_MODEL}")
        print(f"  🎯 Ordinal Regression: {Config.USE_ORDINAL}")
        print(f"  🎨 Prétraitement contraste: {Config.USE_CONTRAST}")
        print(f"  🎲 Mixup: {Config.USE_MIXUP}")
        print(f"  🔄 TTA: {Config.USE_TTA}")
        print(f"  ⚖️  Class Weights: {Config.CLASS_WEIGHTS}")
        print(f"  💾 VRAM attendu: ~7-7.5GB / 8GB")
        print(f"  🎯 Objectif: 88-89%!")
        print("="*70 + "\n")

# ==================== Ordinal Loss ====================
class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits, labels):
        batch_size = labels.size(0)
        device = labels.device
        
        targets = torch.zeros(batch_size, self.num_classes - 1, device=device)
        for i in range(self.num_classes - 1):
            targets[:, i] = (labels > i).float()
        
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

class WeightedOrdinalLoss(nn.Module):
    def __init__(self, num_classes=5, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.ordinal_loss = OrdinalRegressionLoss(num_classes)
    
    def forward(self, logits, labels):
        loss = self.ordinal_loss(logits, labels)
        
        if self.class_weights is not None:
            weights = self.class_weights[labels].to(loss.device)
            loss = (loss * weights.mean()).mean()
        
        return loss

# ==================== Mixup ====================
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== Modèle ====================
class SwinBase384Optimized(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, use_ordinal=True):
        super(SwinBase384Optimized, self).__init__()
        
        self.num_classes = num_classes
        self.use_ordinal = use_ordinal
        
        self.swin = timm.create_model(
            Config.SWIN_MODEL,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            swin_features = self.swin(dummy)
            swin_dim = swin_features.shape[1]
        
        if use_ordinal:
            self.classifier = nn.Sequential(
                nn.Linear(swin_dim, Config.HIDDEN_DIM),
                nn.BatchNorm1d(Config.HIDDEN_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(Config.DROPOUT),
                nn.Linear(Config.HIDDEN_DIM, num_classes - 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(swin_dim, Config.HIDDEN_DIM),
                nn.BatchNorm1d(Config.HIDDEN_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(Config.DROPOUT),
                nn.Linear(Config.HIDDEN_DIM, num_classes)
            )
        
    def forward(self, x):
        features = self.swin(x)
        logits = self.classifier(features)
        return logits
    
    def predict(self, x):
        logits = self.forward(x)
        
        if self.use_ordinal:
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).sum(dim=1)
            return predictions
        else:
            return logits.argmax(dim=1)

# ==================== Dataset ====================
class APTOSDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, use_contrast=False, contrast_factor=1.5):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.use_contrast = use_contrast
        self.contrast_factor = contrast_factor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'id_code']
        
        img_path = os.path.join(self.data_dir, f"{img_name}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_dir, f"{img_name}_m3.png")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.use_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.contrast_factor)
        
        label = int(self.df.loc[idx, 'diagnosis'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==================== Training avec Gradient Accumulation ====================
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Mixup
        if Config.USE_MIXUP and np.random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            
            loss = loss / Config.ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            
            with torch.no_grad():
                if Config.USE_ORDINAL:
                    preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
                else:
                    preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        else:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            loss = loss / Config.ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            
            with torch.no_grad():
                if Config.USE_ORDINAL:
                    preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
                else:
                    preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Gradient Accumulation
        if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Libérer mémoire GPU
            torch.cuda.empty_cache()
        
        running_loss += loss.item() * Config.ACCUMULATION_STEPS
        current_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        
        pbar.set_postfix({
            'loss': f'{loss.item() * Config.ACCUMULATION_STEPS:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    # Dernier step si nécessaire
    if (batch_idx + 1) % Config.ACCUMULATION_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    
    return epoch_loss, epoch_acc, mae

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
            if Config.USE_ORDINAL:
                preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
            else:
                preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            current_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
            # Libérer mémoire
            del images, labels, logits
            torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    
    return epoch_loss, epoch_acc, mae

# ==================== TTA ====================
def predict_with_tta(model, images, device):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        # Original
        with torch.cuda.amp.autocast():
            logits = model(images)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        
        # Horizontal flip
        with torch.cuda.amp.autocast():
            logits = model(torch.flip(images, dims=[3]))
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        
        # Vertical flip
        with torch.cuda.amp.autocast():
            logits = model(torch.flip(images, dims=[2]))
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        
        # Both flips
        with torch.cuda.amp.autocast():
            logits = model(torch.flip(images, dims=[2, 3]))
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
    
    avg_probs = np.mean(all_preds, axis=0)
    
    if Config.USE_ORDINAL:
        preds = (avg_probs > 0.5).sum(axis=1)
    else:
        preds = avg_probs.argmax(axis=1)
    
    return preds

# ==================== Evaluation ====================
def evaluate_model(model, loader, device, use_tta=True):
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc='Testing with TTA' if use_tta else 'Testing'):
        images = images.to(device)
        
        if use_tta and Config.USE_TTA:
            preds = predict_with_tta(model, images, device)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = model(images)
                
                if Config.USE_ORDINAL:
                    preds = (torch.sigmoid(logits) > 0.5).sum(dim=1).cpu().numpy()
                else:
                    preds = logits.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        
        # Libérer mémoire
        del images
        torch.cuda.empty_cache()
    
    return np.array(all_labels), np.array(all_preds)

# ==================== Main ====================
def main():
    Config.print_config()
    
    # Vérifier GPU
    print(f"🔍 GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Charger données
    print("\n📂 Chargement des données...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"  ✅ {len(df)} images chargées")
    
    # Split
    train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df['diagnosis'])
    val_df, test_df = train_test_split(temp_df, test_size=0.6, random_state=42, stratify=temp_df['diagnosis'])
    
    print(f"  📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = APTOSDataset(train_df, Config.DATA_DIR, train_transform, 
                                use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    val_dataset = APTOSDataset(val_df, Config.DATA_DIR, val_transform,
                              use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    test_dataset = APTOSDataset(test_df, Config.DATA_DIR, val_transform,
                               use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                           num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Modèle
    print("\n🔨 Construction du modèle...")
    model = SwinBase384Optimized(
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        use_ordinal=Config.USE_ORDINAL
    )
    model = model.to(Config.DEVICE)
    
    # Loss
    criterion = WeightedOrdinalLoss(
        num_classes=Config.NUM_CLASSES,
        class_weights=Config.CLASS_WEIGHTS.to(Config.DEVICE)
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Training
    print("\n🚀 Début de l'entraînement...")
    
    best_val_mae = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_mae': [],
        'val_loss': [], 'val_acc': [], 'val_mae': []
    }
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Époque [{epoch+1}/{Config.NUM_EPOCHS}]")
        print(f"{'='*70}")
        
        train_loss, train_acc, train_mae = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, scaler)
        val_loss, val_acc, val_mae = validate(model, val_loader, criterion, Config.DEVICE)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mae'].append(val_mae)
        
        print(f"\n📊 Résultats Époque {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train MAE: {train_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val MAE: {val_mae:.4f}")
        
        if val_mae < best_val_mae - Config.MIN_DELTA:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  💾 Modèle sauvegardé (Val MAE: {val_mae:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{Config.PATIENCE}")
            
            if patience_counter >= Config.PATIENCE:
                print(f"\n⏹️  Early stopping à l'époque {epoch+1}")
                break
        
        # Libérer mémoire
        gc.collect()
        torch.cuda.empty_cache()
    
    # Charger meilleur modèle
    print("\n📥 Chargement du meilleur modèle...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    
    # Evaluation AVEC TTA
    print("\n🧪 Évaluation finale sur test set AVEC TTA...")
    test_labels, test_preds = evaluate_model(model, test_loader, Config.DEVICE, use_tta=True)
    
    # Métriques
    overall_acc = accuracy_score(test_labels, test_preds) * 100
    mae = np.mean(np.abs(test_preds - test_labels))
    qwk = cohen_kappa_score(test_labels, test_preds, weights='quadratic')
    
    print(f"\n{'='*70}")
    print(f"📊 RÉSULTATS FINAUX (GPU 8GB + TTA)")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"MAE: {mae:.4f}")
    print(f"Quadratic Kappa: {qwk:.4f}")
    
    if overall_acc >= 89.0:
        print(f"\n🏆🏆🏆 OBJECTIF ATTEINT! {overall_acc:.2f}% >= 89%!")
    elif overall_acc >= 87.0:
        print(f"\n🎯 EXCELLENT! {overall_acc:.2f}% (proche objectif)")
    elif overall_acc >= 85.0:
        print(f"\n✅ TRÈS BON! {overall_acc:.2f}%")
    
    print("\nPar classe:")
    print(classification_report(test_labels, test_preds, 
                               target_names=[f'Classe {i}' for i in range(Config.NUM_CLASSES)]))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Swin 384 GPU 8GB + TTA (Acc: {overall_acc:.1f}%, MAE: {mae:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_gpu8gb.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix: confusion_matrix_gpu8gb.png")
    
    print(f"\n{'='*70}")
    print("✅ Entraînement terminé!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
