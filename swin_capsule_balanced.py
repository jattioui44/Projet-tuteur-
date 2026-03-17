"""
Architecture Hybride Swin + Capsule - VERSION AMÉLIORÉE
AMÉLIORATIONS:
1. Class Weights (contre déséquilibre)
2. Poids équilibrés Swin/Capsules (0.5/0.5)
3. Focal Loss option
4. Plus de régularisation
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
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== Configuration ====================
class Config:
    # Chemins
    DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_outputs/m3"  # Dossier contenant les images 
    CSV_PATH = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train.csv" 
    
    # Hyperparamètres
    IMG_SIZE = 224
    BATCH_SIZE = 12
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-5
    NUM_CLASSES = 5
    NUM_WORKERS = 4
    
    # Modèle
    MODEL_NAME = 'swin_tiny_patch4_window7_224'
    
    # Capsule Network
    NUM_PRIMARY_CAPS = 32
    PRIMARY_CAP_DIM = 8
    NUM_ROUTING_CAPS = 5
    ROUTING_CAP_DIM = 16
    NUM_ROUTING_ITER = 3
    
    # Architecture
    FUSION_DIM = 256
    
    # Early Stopping
    PATIENCE = 10  # Augmenté
    MIN_DELTA = 0.001
    
    # Régularisation AUGMENTÉE
    DROPOUT = 0.4  # Augmenté de 0.3 à 0.4
    DROP_PATH_RATE = 0.3  # Augmenté de 0.2 à 0.3
    WEIGHT_DECAY = 0.15  # Augmenté de 0.1 à 0.15
    LABEL_SMOOTHING = 0.1
    
    # Loss weights - ÉQUILIBRÉS! ⭐
    CAPSULE_LOSS_WEIGHT = 0.5  # Réduit de 0.7 à 0.5
    SWIN_LOSS_WEIGHT = 0.5     # Augmenté de 0.3 à 0.5
    
    # Class Weights - NOUVEAU! ⭐
    USE_CLASS_WEIGHTS = True
    
    # Focal Loss - NOUVEAU! ⭐
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Sauvegarde
    BEST_MODEL_PATH = "./best_swin_capsule_balanced.pth"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre des classes
    Focus sur les exemples difficiles
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==================== Capsule Components ====================
def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-8)

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, num_capsules, cap_dim, kernel_size=9, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.cap_dim = cap_dim
        
        self.conv = nn.Conv2d(
            in_channels, 
            num_capsules * cap_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        
    def forward(self, x):
        output = self.conv(x)
        batch_size = output.shape[0]
        output = output.view(batch_size, self.num_capsules, self.cap_dim, -1)
        output = output.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch_size, -1, self.cap_dim)
        output = squash(output, dim=-1)
        return output

class RoutingCapsules(nn.Module):
    def __init__(self, num_in_caps, in_cap_dim, num_out_caps, out_cap_dim, num_iterations=3):
        super(RoutingCapsules, self).__init__()
        self.num_in_caps = num_in_caps
        self.num_out_caps = num_out_caps
        self.num_iterations = num_iterations
        
        self.W = nn.Parameter(
            torch.randn(1, num_in_caps, num_out_caps, out_cap_dim, in_cap_dim)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(2).unsqueeze(4)
        W_expanded = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W_expanded, x_expanded).squeeze(-1)
        
        b = torch.zeros(batch_size, self.num_in_caps, self.num_out_caps, 1).to(x.device)
        
        for iteration in range(self.num_iterations):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1)
            v = squash(s, dim=-1)
            
            if iteration < self.num_iterations - 1:
                v_expanded = v.unsqueeze(1)
                agreement = (u_hat * v_expanded).sum(dim=-1, keepdim=True)
                b = b + agreement
        
        return v

class CapsuleNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CapsuleNetwork, self).__init__()
        
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 256 * 6 * 6),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        self.primary_caps = PrimaryCapsules(
            in_channels=256,
            num_capsules=Config.NUM_PRIMARY_CAPS,
            cap_dim=Config.PRIMARY_CAP_DIM,
            kernel_size=3,
            stride=1
        )
        
        num_primary_caps_total = Config.NUM_PRIMARY_CAPS * 16
        
        self.routing_caps = RoutingCapsules(
            num_in_caps=num_primary_caps_total,
            in_cap_dim=Config.PRIMARY_CAP_DIM,
            num_out_caps=num_classes,
            out_cap_dim=Config.ROUTING_CAP_DIM,
            num_iterations=Config.NUM_ROUTING_ITER
        )
        
    def forward(self, x):
        x = self.feature_projection(x)
        x = x.view(-1, 256, 6, 6)
        x = self.primary_caps(x)
        x = self.routing_caps(x)
        class_probs = torch.sqrt((x ** 2).sum(dim=-1))
        return class_probs, x

# ==================== Architecture Hybride ====================
class SwinCapsuleHybrid(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(SwinCapsuleHybrid, self).__init__()
        
        self.swin = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=Config.DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            swin_features = self.swin(dummy)
            swin_feature_dim = swin_features.shape[1]
        
        print(f"✅ Swin dim: {swin_feature_dim}")
        
        self.capsule_net = CapsuleNetwork(swin_feature_dim, num_classes)
        
        self.swin_classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(swin_feature_dim, num_classes)
        )
        
        fusion_input_dim = swin_feature_dim + (num_classes * Config.ROUTING_CAP_DIM)
        
        # Fusion layer plus profonde
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_DIM),
            nn.BatchNorm1d(Config.FUSION_DIM),  # Ajouté
            nn.ReLU(),
            nn.Dropout(0.5),  # Augmenté
            nn.Linear(Config.FUSION_DIM, Config.FUSION_DIM // 2),
            nn.BatchNorm1d(Config.FUSION_DIM // 2),  # Ajouté
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(Config.FUSION_DIM // 2, num_classes)
        )
        
    def forward(self, x):
        swin_features = self.swin(x)
        swin_logits = self.swin_classifier(swin_features)
        capsule_probs, capsule_vectors = self.capsule_net(swin_features)
        
        capsule_flat = capsule_vectors.view(capsule_vectors.size(0), -1)
        fusion_input = torch.cat([swin_features, capsule_flat], dim=1)
        fusion_logits = self.fusion_layer(fusion_input)
        
        return fusion_logits, swin_logits, capsule_probs

# ==================== Loss Hybride ====================
class HybridLoss(nn.Module):
    def __init__(self, class_weights=None, use_focal=False):
        super(HybridLoss, self).__init__()
        
        if use_focal:
            print("✅ Utilisation de Focal Loss")
            self.ce_loss = FocalLoss(
                alpha=Config.FOCAL_ALPHA,
                gamma=Config.FOCAL_GAMMA
            )
        else:
            print("✅ Utilisation de CrossEntropy avec class weights")
            self.ce_loss = nn.CrossEntropyLoss(
                label_smoothing=Config.LABEL_SMOOTHING,
                weight=class_weights
            )
        
        self.class_weights = class_weights
        
    def margin_loss(self, capsule_probs, labels):
        batch_size = capsule_probs.size(0)
        labels_one_hot = F.one_hot(labels, num_classes=Config.NUM_CLASSES).float()
        
        loss_positive = labels_one_hot * F.relu(0.9 - capsule_probs) ** 2
        loss_negative = 0.5 * (1 - labels_one_hot) * F.relu(capsule_probs - 0.1) ** 2
        
        loss = (loss_positive + loss_negative).sum(dim=1)
        
        # Appliquer class weights si disponibles
        if self.class_weights is not None:
            weights = self.class_weights[labels]
            loss = loss * weights
        
        return loss.mean()
    
    def forward(self, fusion_logits, swin_logits, capsule_probs, labels):
        loss_fusion = self.ce_loss(fusion_logits, labels)
        loss_swin = self.ce_loss(swin_logits, labels)
        loss_capsule = self.margin_loss(capsule_probs, labels)
        
        # Poids ÉQUILIBRÉS maintenant! ⭐
        total_loss = (
            loss_fusion + 
            Config.SWIN_LOSS_WEIGHT * loss_swin + 
            Config.CAPSULE_LOSS_WEIGHT * loss_capsule
        )
        
        return total_loss, loss_fusion, loss_swin, loss_capsule

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
    print("\n📊 Distribution des classes:")
    class_dist = df['diagnosis'].value_counts().sort_index()
    print(class_dist)
    
    # Calcul class weights ⭐
    if Config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df['diagnosis']),
            y=df['diagnosis']
        )
        class_weights = torch.FloatTensor(class_weights).to(Config.DEVICE)
        print("\n⚖️  Class Weights calculés:")
        for i, w in enumerate(class_weights):
            print(f"   Classe {i}: {w:.3f} (count: {class_dist[i]})")
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights

# ==================== Transformations ====================
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),  # Augmenté
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # Augmenté
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Nouveau
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
        
        loss, loss_f, loss_s, loss_c = criterion(fusion_logits, swin_logits, capsule_probs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = fusion_logits.max(1)
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
            
            fusion_logits, swin_logits, capsule_probs = model(images)
            loss, _, _, _ = criterion(fusion_logits, swin_logits, capsule_probs, labels)
            
            running_loss += loss.item()
            _, predicted = fusion_logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), 100. * accuracy_score(all_labels, all_preds), all_preds, all_labels

# ==================== Main ====================
def main():
    print("\n" + "="*70)
    print("🚀 SWIN + CAPSULE - VERSION AMÉLIORÉE")
    print("="*70)
    print("AMÉLIORATIONS:")
    print("✅ Class Weights (déséquilibre)")
    print("✅ Poids équilibrés: Swin 0.5 / Capsule 0.5")
    print(f"✅ Focal Loss: {Config.USE_FOCAL_LOSS}")
    print(f"✅ Régularisation augmentée")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ GPU non disponible!")
        return
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Données
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = prepare_data()
    
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
    model = SwinCapsuleHybrid(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(Config.DEVICE)
    
    # Loss avec class weights
    criterion = HybridLoss(class_weights=class_weights, use_focal=Config.USE_FOCAL_LOSS)
    
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
    
    # Visualisations
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion (Balanced)')
    plt.ylabel('Vraie')
    plt.xlabel('Prédite')
    plt.savefig('confusion_matrix_balanced.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Loss (Balanced)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Val', marker='s')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy (Balanced)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_balanced.png', dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print(f"🏆 Best val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"🧪 Test acc: {test_acc:.2f}%")
    print("\n🎉 Terminé!")

if __name__ == "__main__":
    main()
