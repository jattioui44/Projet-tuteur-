"""
MODÈLE HYBRIDE: SWIN TRANSFORMER + CONVNEXT
────────────────────────────────────────────

Architecture à deux branches complémentaires:
• Branche 1: Swin Transformer Tiny 224 (attention globale)
• Branche 2: ConvNeXt Tiny 224 (patterns convolutionnels)
• Fusion: Concatenation features

Avantages:
• Transformer + CNN = Complémentarité maximale
• 224×224 = GPU 8GB friendly
• Batch 16 = Stable
• Régularisation AJUSTÉE (fix échec précédent)

Objectif: 86-88%
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
    IMG_SIZE = 224
    BATCH_SIZE = 14  # Légèrement réduit (2 branches)
    
    # ==================== TRAINING ====================
    NUM_EPOCHS = 100
    LEARNING_RATE = 3.0e-5  # ⬆️ Augmenté
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # ==================== MODÈLES ====================
    SWIN_MODEL = 'swin_tiny_patch4_window7_224'  # Transformer
    CONVNEXT_MODEL = 'convnext_tiny'  # CNN
    HIDDEN_DIM = 512
    
    # ==================== RÉGULARISATION (AJUSTÉE!) ====================
    DROPOUT = 0.3  # ⬇️ Réduit de 0.5 (fix train acc bas)
    DROP_PATH_RATE = 0.2  # ⬇️ Réduit de 0.4
    WEIGHT_DECAY = 0.1  # ⬇️ Réduit de 0.2
    
    # ==================== ORDINAL REGRESSION ====================
    USE_ORDINAL = True
    
    # ==================== CLASS WEIGHTS (AJUSTÉS!) ====================
    CLASS_WEIGHTS = torch.tensor([0.3, 1.8, 1.0, 5.0, 2.2])  # ⬆️ Classe 3: 5.0!
    
    # ==================== PRÉTRAITEMENT ====================
    USE_CONTRAST = True
    CONTRAST_FACTOR = 1.5
    
    # ==================== MIXUP ====================
    USE_MIXUP = True
    MIXUP_ALPHA = 0.3
    
    # ==================== TEST-TIME AUGMENTATION ====================
    USE_TTA = True
    TTA_TRANSFORMS = 4
    
    # ==================== MIXED PRECISION ====================
    USE_MIXED_PRECISION = True
    
    # ==================== EARLY STOPPING ====================
    PATIENCE = 30
    MIN_DELTA = 0.001
    
    # ==================== SAUVEGARDE ====================
    BEST_MODEL_PATH = "./best_hybrid_swin_convnext224.pth"
    HISTORY_PATH = "./history_hybrid_swin_convnext224.npy"
    
    # ==================== DEVICE ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def print_config():
        print("\n" + "="*70)
        print("📋 CONFIGURATION HYBRIDE SWIN + CONVNEXT 224")
        print("="*70)
        print(f"  📸 Image size: {Config.IMG_SIZE}×{Config.IMG_SIZE}")
        print(f"  📦 Batch size: {Config.BATCH_SIZE}")
        print(f"  🧠 Branche 1 (Transformer): {Config.SWIN_MODEL}")
        print(f"  🧠 Branche 2 (CNN): {Config.CONVNEXT_MODEL}")
        print(f"  🎯 Ordinal Regression: {Config.USE_ORDINAL}")
        print(f"  🎨 Prétraitement contraste: {Config.USE_CONTRAST}")
        print(f"  🎲 Mixup: {Config.USE_MIXUP}")
        print(f"  🔄 TTA: {Config.USE_TTA}")
        print(f"  ⚖️  Class Weights: {Config.CLASS_WEIGHTS}")
        print(f"  📈 Learning Rate: {Config.LEARNING_RATE}")
        print(f"  💾 VRAM attendu: ~6-7GB / 8GB")
        print(f"  🎯 Objectif: 86-88%!")
        print("")
        print("  ✨ ARCHITECTURE HYBRIDE:")
        print("     • Swin Tiny: Attention globale (Transformer)")
        print("     • ConvNeXt Tiny: Patterns locaux (CNN)")
        print("     • Complémentarité Transformer + CNN")
        print("")
        print("  🔧 CORRECTIONS vs ConvNeXt échec:")
        print("     • Dropout: 0.5 → 0.3 (fix train acc bas)")
        print("     • Drop Path: 0.4 → 0.2")
        print("     • Weight Decay: 0.2 → 0.1")
        print("     • Classe 3 weight: 3.0 → 5.0 (fix 27.6%)")
        print("     • LR: 2.0e-5 → 3.0e-5")
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
def mixup_data(x, y, alpha=0.3):
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

# ==================== Modèle Hybride Swin + ConvNeXt ====================
class HybridSwinConvNeXt(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, use_ordinal=True):
        super(HybridSwinConvNeXt, self).__init__()
        
        self.num_classes = num_classes
        self.use_ordinal = use_ordinal
        
        # ==================== BRANCHE 1: SWIN TRANSFORMER (ATTENTION) ====================
        print("\n🔨 Chargement Swin Transformer Tiny...")
        self.swin = timm.create_model(
            Config.SWIN_MODEL,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # ==================== BRANCHE 2: CONVNEXT (CNN) ====================
        print("🔨 Chargement ConvNeXt Tiny...")
        self.convnext = timm.create_model(
            Config.CONVNEXT_MODEL,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            swin_features = self.swin(dummy)
            convnext_features = self.convnext(dummy)
            
            swin_dim = swin_features.shape[1]
            convnext_dim = convnext_features.shape[1]
            total_dim = swin_dim + convnext_dim
            
            print(f"\n🔍 Feature Dimensions:")
            print(f"  Swin Transformer: {swin_dim}")
            print(f"  ConvNeXt: {convnext_dim}")
            print(f"  Total (concatenated): {total_dim}\n")
        
        # ==================== FUSION & CLASSIFIER ====================
        if use_ordinal:
            self.classifier = nn.Sequential(
                nn.Linear(total_dim, Config.HIDDEN_DIM),
                nn.BatchNorm1d(Config.HIDDEN_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(Config.DROPOUT),
                nn.Linear(Config.HIDDEN_DIM, num_classes - 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(total_dim, Config.HIDDEN_DIM),
                nn.BatchNorm1d(Config.HIDDEN_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(Config.DROPOUT),
                nn.Linear(Config.HIDDEN_DIM, num_classes)
            )
        
    def forward(self, x):
        # Branche 1: Swin Transformer (attention globale)
        swin_features = self.swin(x)
        
        # Branche 2: ConvNeXt (patterns convolutionnels)
        convnext_features = self.convnext(x)
        
        # Fusion: Concatenation
        fused_features = torch.cat([swin_features, convnext_features], dim=1)
        
        # Classification
        logits = self.classifier(fused_features)
        
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

# ==================== Training ====================
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixup
        if Config.USE_MIXUP and np.random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                if Config.USE_ORDINAL:
                    preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
                else:
                    preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        running_loss += loss.item()
        current_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })
        
        # Libérer mémoire
        torch.cuda.empty_cache()
    
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
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    
    return epoch_loss, epoch_acc, mae

# ==================== TTA ====================
def predict_with_tta(model, images, device):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = model(images)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        
        with torch.cuda.amp.autocast():
            logits = model(torch.flip(images, dims=[3]))
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        
        with torch.cuda.amp.autocast():
            logits = model(torch.flip(images, dims=[2]))
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        
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
    
    return np.array(all_labels), np.array(all_preds)

# ==================== Main ====================
def main():
    Config.print_config()
    
    print(f"🔍 GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n📂 Chargement des données...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"  ✅ {len(df)} images chargées")
    
    train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df['diagnosis'])
    val_df, test_df = train_test_split(temp_df, test_size=0.6, random_state=42, stratify=temp_df['diagnosis'])
    
    print(f"  📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = APTOSDataset(train_df, Config.DATA_DIR, train_transform, 
                                use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    val_dataset = APTOSDataset(val_df, Config.DATA_DIR, val_transform,
                              use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    test_dataset = APTOSDataset(test_df, Config.DATA_DIR, val_transform,
                               use_contrast=Config.USE_CONTRAST, contrast_factor=Config.CONTRAST_FACTOR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                           num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    print("\n🔨 Construction du modèle hybride Swin + ConvNeXt...")
    model = HybridSwinConvNeXt(
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        use_ordinal=Config.USE_ORDINAL
    )
    model = model.to(Config.DEVICE)
    
    criterion = WeightedOrdinalLoss(
        num_classes=Config.NUM_CLASSES,
        class_weights=Config.CLASS_WEIGHTS.to(Config.DEVICE)
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    scaler = torch.cuda.amp.GradScaler()
    
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
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Sauvegarder historique
    print("\n💾 Sauvegarde de l'historique...")
    np.save(Config.HISTORY_PATH, history)
    print(f"  ✅ Historique sauvegardé: {Config.HISTORY_PATH}")
    
    print("\n📥 Chargement du meilleur modèle...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    
    print("\n🧪 Évaluation finale sur test set AVEC TTA...")
    test_labels, test_preds = evaluate_model(model, test_loader, Config.DEVICE, use_tta=True)
    
    overall_acc = accuracy_score(test_labels, test_preds) * 100
    mae = np.mean(np.abs(test_preds - test_labels))
    qwk = cohen_kappa_score(test_labels, test_preds, weights='quadratic')
    
    print(f"\n{'='*70}")
    print(f"📊 RÉSULTATS FINAUX (HYBRIDE SWIN+CONVNEXT + TTA)")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"MAE: {mae:.4f}")
    print(f"Quadratic Kappa: {qwk:.4f}")
    
    if overall_acc >= 88.0:
        print(f"\n🏆 EXCELLENT! {overall_acc:.2f}% >= 88%!")
    elif overall_acc >= 86.0:
        print(f"\n🎯 TRÈS BON! {overall_acc:.2f}% >= 86%!")
    elif overall_acc >= 84.0:
        print(f"\n✅ BON! {overall_acc:.2f}% >= 84%!")
    else:
        print(f"\n⚠️ Besoin amélioration: {overall_acc:.2f}%")
    
    print("\nPar classe:")
    print(classification_report(test_labels, test_preds, 
                               target_names=[f'Classe {i}' for i in range(Config.NUM_CLASSES)]))
    
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Hybrid Swin+ConvNeXt + TTA (Acc: {overall_acc:.1f}%, MAE: {mae:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_hybrid_swin_convnext.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix: confusion_matrix_hybrid_swin_convnext.png")
    
    # Training curves
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training & Validation MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_hybrid_swin_convnext.png', dpi=300, bbox_inches='tight')
    print(f"✅ Training curves: training_curves_hybrid_swin_convnext.png")
    
    print(f"\n{'='*70}")
    print("✅ Entraînement terminé!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
