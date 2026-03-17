"""
Architecture PARALLÈLE ADVANCED
Toutes les astuces: Focal Loss + Ordinal + Mixup + TTA + Warm Restarts

FOCUS: Résultats globaux + Classe 3
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

# ==================== Configuration ====================
class Config:
    # Chemins
    DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_images"  # Dossier contenant les images 
    CSV_PATH = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train.csv"
    
    # Hyperparamètres
    IMG_SIZE = 224
    BATCH_SIZE = 12
    NUM_EPOCHS = 100
    LEARNING_RATE = 3e-5
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Modèles
    SWIN_MODEL = 'swin_tiny_patch4_window7_224'
    CONVNEXT_MODEL = 'convnext_tiny'
    
    # ==================== CLASSIFIER PROFOND ====================
    HIDDEN_DIMS = [1024, 512, 256, 128, 32]
    DROPOUT = 0.3  # CORRIGÉ! (était 0.5)
    
    # Régularisation
    DROP_PATH_RATE = 0.1
    WEIGHT_DECAY = 0.1
    
    # ==================== ASTUCES AVANCÉES ====================
    # Focal Loss
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 1.0
    FOCAL_GAMMA = 2.0  # Focus sur exemples difficiles
    
    # Ordinal Regression
    USE_ORDINAL = True
    
    # Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    
    # TTA (Test-Time Augmentation)
    USE_TTA = True
    TTA_TRANSFORMS = 4
    
    # Warm Restarts
    USE_WARM_RESTARTS = True
    T_0 = 10  # Redémarre tous les 10 epochs
    T_MULT = 2
    
    # Loss weights
    SWIN_WEIGHT = 0.5
    CONVNEXT_WEIGHT = 0.5
    
    # Prétraitement
    USE_CONTRAST = True
    CONTRAST_FACTOR = 1.5
    
    # Early Stopping
    PATIENCE = 8
    MIN_DELTA = 0.001
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_parallel_ADVANCED.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    """
    Focal Loss - réduit poids des exemples faciles
    Force le modèle sur exemples difficiles (Classe 3!)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==================== Ordinal Regression ====================
class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression - utilise ordre naturel 0<1<2<3<4
    Erreur 2→4 moins grave que 0→4
    """
    def __init__(self, num_classes=5):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, logits, targets):
        # Logits: [batch, num_classes]
        # Targets: [batch]
        
        batch_size = logits.size(0)
        
        # Créer labels ordinaux [batch, num_classes-1]
        # Pour classe k, on veut [1,1,...,1,0,0,...,0] avec k fois 1
        ordinal_labels = torch.zeros(batch_size, self.num_classes - 1, device=logits.device)
        for i in range(batch_size):
            if targets[i] > 0:
                ordinal_labels[i, :targets[i]] = 1
        
        # Prédictions ordinales (sigmoid sur cumsum)
        ordinal_preds = torch.sigmoid(logits[:, :-1])
        
        # BCE loss
        loss = F.binary_cross_entropy(ordinal_preds, ordinal_labels)
        
        return loss

# ==================== Mixup ====================
def mixup_data(x, y, alpha=0.4):
    """
    Mixup - mélange images et labels
    Crée exemples synthétiques
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss pour mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== Modèle Parallèle ADVANCED ====================
class ParallelSwinConvNeXtAdvanced(nn.Module):
    """
    Architecture parallèle avec classifier profond
    + Support ordinal regression
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(ParallelSwinConvNeXtAdvanced, self).__init__()
        
        self.num_classes = num_classes
        
        print("\n🔨 Construction architecture parallèle ADVANCED...")
        
        # Branches
        print("  📦 Branche 1: Swin Transformer Tiny")
        self.swin = timm.create_model(
            Config.SWIN_MODEL,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        print("  📦 Branche 2: ConvNeXt Tiny")
        self.convnext = timm.create_model(
            Config.CONVNEXT_MODEL,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        # Get dimensions
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
        
        # Classifier profond
        print(f"\n  🏗️  CLASSIFIER PROFOND:")
        print(f"     Architecture: {fused_dim} → {' → '.join(map(str, Config.HIDDEN_DIMS))} → {num_classes}")
        print(f"     Dropout: {Config.DROPOUT}")
        
        layers = []
        input_dim = fused_dim
        
        for hidden_dim in Config.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(Config.DROPOUT)
            ])
            input_dim = hidden_dim
        
        # Couche finale (num_classes pour ordinal, num_classes-1 pour ordinal regression)
        if Config.USE_ORDINAL:
            layers.append(nn.Linear(input_dim, num_classes - 1))
            print(f"  🔢 Ordinal Regression activé")
        else:
            layers.append(nn.Linear(input_dim, num_classes))
        
        self.fusion_classifier = nn.Sequential(*layers)
        
        # Classifiers auxiliaires
        self.swin_classifier = nn.Linear(swin_dim, num_classes if not Config.USE_ORDINAL else num_classes - 1)
        self.convnext_classifier = nn.Linear(convnext_dim, num_classes if not Config.USE_ORDINAL else num_classes - 1)
        
        # Astuces
        print(f"\n  🎯 ASTUCES ACTIVÉES:")
        if Config.USE_FOCAL_LOSS:
            print(f"     ✅ Focal Loss (gamma={Config.FOCAL_GAMMA})")
        if Config.USE_ORDINAL:
            print(f"     ✅ Ordinal Regression")
        if Config.USE_MIXUP:
            print(f"     ✅ Mixup (alpha={Config.MIXUP_ALPHA})")
        if Config.USE_TTA:
            print(f"     ✅ TTA ({Config.TTA_TRANSFORMS} transforms)")
        if Config.USE_WARM_RESTARTS:
            print(f"     ✅ Warm Restarts (T_0={Config.T_0})")
        print()
    
    def forward(self, x):
        swin_features = self.swin(x)
        swin_logits = self.swin_classifier(swin_features)
        
        convnext_features = self.convnext(x)
        convnext_logits = self.convnext_classifier(convnext_features)
        
        fused_features = torch.cat([swin_features, convnext_features], dim=1)
        fusion_logits = self.fusion_classifier(fused_features)
        
        return fusion_logits, swin_logits, convnext_logits
    
    def predict_ordinal(self, logits):
        """Convertit logits ordinaux en prédictions de classe"""
        # logits: [batch, num_classes-1]
        probs = torch.sigmoid(logits)
        # Cumulative probabilities -> classe prédite
        cumulative = torch.cat([
            torch.ones(probs.size(0), 1, device=probs.device),
            probs
        ], dim=1)
        predicted = (cumulative > 0.5).sum(dim=1) - 1
        return predicted.clamp(0, self.num_classes - 1)

# ==================== Loss Hybride ADVANCED ====================
class HybridAdvancedLoss(nn.Module):
    """Loss avec Focal + Ordinal"""
    def __init__(self):
        super(HybridAdvancedLoss, self).__init__()
        if Config.USE_FOCAL_LOSS:
            self.ce_loss = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        if Config.USE_ORDINAL:
            self.ordinal_loss = OrdinalRegressionLoss(num_classes=Config.NUM_CLASSES)
    
    def forward(self, fusion_logits, swin_logits, convnext_logits, labels):
        if Config.USE_ORDINAL:
            fusion_loss = self.ordinal_loss(fusion_logits, labels)
            swin_loss = self.ordinal_loss(swin_logits, labels)
            convnext_loss = self.ordinal_loss(convnext_logits, labels)
        else:
            fusion_loss = self.ce_loss(fusion_logits, labels)
            swin_loss = self.ce_loss(swin_logits, labels)
            convnext_loss = self.ce_loss(convnext_logits, labels)
        
        total_loss = fusion_loss + Config.SWIN_WEIGHT * swin_loss + Config.CONVNEXT_WEIGHT * convnext_loss
        
        return total_loss, fusion_loss, swin_loss, convnext_loss

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
    
    image_paths = [os.path.join(Config.DATA_DIR, f"{id_code}.png") for id_code in df['id_code']]
    labels = df['diagnosis'].values
    
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

# TTA transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomRotation([90, 90]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]

# ==================== Training ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Mixup
        if Config.USE_MIXUP and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            
            optimizer.zero_grad()
            
            fusion_logits, swin_logits, convnext_logits = model(images)
            
            total_loss_a, _, _, _ = criterion(fusion_logits, swin_logits, convnext_logits, labels_a)
            total_loss_b, _, _, _ = criterion(fusion_logits, swin_logits, convnext_logits, labels_b)
            total_loss = lam * total_loss_a + (1 - lam) * total_loss_b
        else:
            optimizer.zero_grad()
            
            fusion_logits, swin_logits, convnext_logits = model(images)
            
            total_loss, fusion_loss, swin_loss, convnext_loss = criterion(
                fusion_logits, swin_logits, convnext_logits, labels
            )
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        
        # Prédictions
        if Config.USE_ORDINAL:
            predicted = model.predict_ordinal(fusion_logits)
        else:
            _, predicted = fusion_logits.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
        
        del images, labels, fusion_logits, swin_logits, convnext_logits, total_loss
        torch.cuda.empty_cache()
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device, use_tta=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            if use_tta and Config.USE_TTA:
                # TTA: moyenne sur plusieurs augmentations
                tta_preds = []
                for _ in range(Config.TTA_TRANSFORMS):
                    fusion_logits, _, _ = model(images)
                    if Config.USE_ORDINAL:
                        pred = model.predict_ordinal(fusion_logits)
                    else:
                        _, pred = fusion_logits.max(1)
                    tta_preds.append(pred)
                
                # Vote majoritaire
                tta_preds = torch.stack(tta_preds, dim=0)
                predicted = torch.mode(tta_preds, dim=0)[0]
            else:
                fusion_logits, swin_logits, convnext_logits = model(images)
                
                total_loss, _, _, _ = criterion(
                    fusion_logits, swin_logits, convnext_logits, labels
                )
                
                running_loss += total_loss.item()
                
                if Config.USE_ORDINAL:
                    predicted = model.predict_ordinal(fusion_logits)
                else:
                    _, predicted = fusion_logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'acc': f'{100.*accuracy_score(all_labels, all_preds):.2f}%'
            })
            
            del images, labels
            if not use_tta:
                del fusion_logits, swin_logits, convnext_logits
            torch.cuda.empty_cache()
    
    return running_loss / len(dataloader), 100. * accuracy_score(all_labels, all_preds), all_preds, all_labels

# ==================== Main ====================
def main():
    print("\n" + "="*70)
    print("🚀 ARCHITECTURE PARALLÈLE - ADVANCED (TOUTES ASTUCES)")
    print("="*70)
    print("ARCHITECTURE:")
    print("  📦 Branche 1: Swin Transformer Tiny 224")
    print("  📦 Branche 2: ConvNeXt Tiny 224")
    print("  🔀 Fusion: Concatenation [1536]")
    print(f"  🏗️  Classifier: 1536 → {' → '.join(map(str, Config.HIDDEN_DIMS))} → 5")
    print(f"  💧 Dropout: {Config.DROPOUT} (CORRIGÉ!)")
    print("\nASTUCES:")
    if Config.USE_FOCAL_LOSS:
        print(f"  ✅ Focal Loss (gamma={Config.FOCAL_GAMMA})")
    if Config.USE_ORDINAL:
        print("  ✅ Ordinal Regression")
    if Config.USE_MIXUP:
        print(f"  ✅ Mixup (alpha={Config.MIXUP_ALPHA})")
    if Config.USE_TTA:
        print(f"  ✅ TTA ({Config.TTA_TRANSFORMS} transforms)")
    if Config.USE_WARM_RESTARTS:
        print(f"  ✅ Warm Restarts (T_0={Config.T_0})")
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
    model = ParallelSwinConvNeXtAdvanced(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Paramètres totaux: {total_params:,}")
    print(f"📊 Paramètres entraînables: {trainable_params:,}")
    
    # Loss et optimizer
    criterion = HybridAdvancedLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    
    if Config.USE_WARM_RESTARTS:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT
        )
    else:
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
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Test avec TTA
    print("\n" + "="*70)
    print("🧪 ÉVALUATION FINALE (avec TTA)")
    print("="*70)
    
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, Config.DEVICE, use_tta=Config.USE_TTA
    )
    
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    
    # Kappa
    kappa = cohen_kappa_score(test_labels, test_preds, weights='quadratic')
    print(f"📊 Quadratic Kappa: {kappa:.4f}")
    
    if test_acc >= 85.0:
        print(f"\n🏆 EXCELLENT! {test_acc:.2f}% >= 85%!")
    elif test_acc >= 83.0:
        print(f"\n🎯 TRÈS BON! {test_acc:.2f}% >= 83%!")
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
    plt.title(f'Parallel ADVANCED (Acc: {test_acc:.1f}%, Kappa: {kappa:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_parallel_ADVANCED.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix: confusion_matrix_parallel_ADVANCED.png")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss (Parallel ADVANCED)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Parallel ADVANCED)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_parallel_ADVANCED.png', dpi=300, bbox_inches='tight')
    print(f"✅ Training curves: training_curves_parallel_ADVANCED.png")
    
    print("\n" + "="*70)
    print(f"🏆 Meilleur val accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"🧪 Test accuracy: {test_acc:.2f}%")
    print(f"📊 Quadratic Kappa: {kappa:.4f}")
    print("\n🎉 Entraînement terminé!")
    print(f"💾 Modèle sauvegardé: {Config.BEST_MODEL_PATH}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
