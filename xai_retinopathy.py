"""
XAI (Explainable AI) pour Modèle Hybride Swin + Capsules
Rétinopathie Diabétique - APTOS2019

3 méthodes gradient-based:
1. Saliency (Vanilla Gradients)
2. SmoothGrad
3. Integrated Gradients
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
from torchvision import transforms
import timm

import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.activations = None
        self.gradients = None
        
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, target_class):

        x = x.clone().detach()
        x.requires_grad_(True)   # 🔥 IMPORTANT

        self.model.zero_grad()

        logits = self.model(x)

        score = logits[:, target_class]

        score.backward(retain_graph=True)

        grads = self.gradients
        activations = self.activations

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        sum_activations = torch.sum(activations, dim=(2,3), keepdim=True)

        eps = 1e-8

        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / (alpha_denom + eps)

        weights = torch.sum(alphas * F.relu(grads), dim=(2,3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(x.shape[2], x.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        from scipy.ndimage import gaussian_filter
        cam = gaussian_filter(cam, sigma=2)
        return cam
# ==================== Configuration ====================
class Config:
    # Modèle
    MODEL_PATH = "./best_parallel_swin_convnext_simple.pth"  # Changez si besoin
    
    # Données
    DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_images"
    
    # Image à expliquer
    IMAGE_PATH = None  # Sera choisi automatiquement
    IMAGE_ID = "10f10fd30718.png"    # Ou spécifiez un ID (ex: "0a2b8c4d5e.png")
    
    # Paramètres
    IMG_SIZE = 224
    NUM_CLASSES = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Classes rétinopathie
    CLASS_NAMES = [
        "No DR (0)",
        "Mild (1)", 
        "Moderate (2)",
        "Severe (3)",
        "Proliferative DR (4)"
    ]
    '''
    📊 Dataset Statistics:
        Mean: [0.41384357213974, 0.22102037072181702, 0.0736616998910904]
    Std: [0.23784145712852478, 0.13096213340759277, 0.04815368354320526]
    '''
    # XAI
    USE_CONTRAST = True
    CONTRAST_FACTOR = 1.5
    
    # Output
    OUTPUT_DIR = "./xai_outputs"

# ==================== Architecture Modèle ====================
class ParallelSwinConvNeXt(nn.Module):
    """Architecture parallèle Swin + ConvNeXt - VERSION CHECKPOINT"""
    def __init__(self, num_classes=5):
        super(ParallelSwinConvNeXt, self).__init__()
        
        # Swin Transformer
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1
        )
        
        # ConvNeXt
        self.convnext = timm.create_model(
            'convnext_tiny',
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1
        )
        
        # Alpha (poids adaptatif)
        self.alpha = nn.Parameter(torch.tensor(0.3))
        
        # Fusion classifier (3 hidden layers: 1024, 512, 256)
        self.fusion_classifier = nn.Sequential(
            # Layer 0-3: 1536 → 1024
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # Layer 4-7: 1024 → 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # Layer 8-10: 512 → 256 (PAS de Dropout!)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Layer 11: 256 → 5 (finale)
            nn.Linear(256, num_classes)
        )
        
        # Classifiers auxiliaires (768 → 256 → 5) AVEC Dropout!
        self.swin_classifier = nn.Sequential(
            nn.Linear(768, 256),      # Layer 0
            nn.ReLU(inplace=True),    # Layer 1
            nn.Dropout(0.3),          # Layer 2
            nn.Linear(256, num_classes)  # Layer 3
        )
        
        self.convnext_classifier = nn.Sequential(
            nn.Linear(768, 256),      # Layer 0
            nn.ReLU(inplace=True),    # Layer 1
            nn.Dropout(0.3),          # Layer 2
            nn.Linear(256, num_classes)  # Layer 3
        )
    
    def forward(self, x):
        swin_features = self.swin(x)
        convnext_features = self.convnext(x)
        
        fused_features = torch.cat([swin_features, convnext_features], dim=1)
        fusion_logits = self.fusion_classifier(fused_features)
        
        # Pour XAI, on retourne seulement fusion_logits
        return fusion_logits
    
# ==================== Méthodes XAI ====================

def explain_saliency(model, x, target_idx, device):
    """
    Saliency (Vanilla Gradients)
    Calcule les gradients de l'input
    """
    model.eval()
    x = x.clone().requires_grad_(True)
    
    # Forward
    logits = model(x)
    
    # Backward sur classe target
    model.zero_grad()
    logits[0, target_idx].backward()
    
    # Gradients
    grads = x.grad.data.abs()
    
    # Agrégation sur canaux RGB
    saliency = torch.max(grads[0], dim=0)[0].cpu().numpy()
    
    # Normalisation
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency

def explain_smoothgrad(model, x, target_idx, device, n_samples=50, noise_sigma=0.05):
    """
    SmoothGrad
    Moyenne des gradients sur versions bruitées
    """
    model.eval()
    
    saliency_maps = []
    
    for _ in range(n_samples):
        # Ajouter bruit
        noise = torch.randn_like(x) * noise_sigma
        x_noisy = x + noise
        x_noisy = x_noisy.clone().requires_grad_(True)
        
        # Forward
        logits = model(x_noisy)
        
        # Backward
        model.zero_grad()
        logits[0, target_idx].backward()
        
        # Gradients
        grads = x_noisy.grad.data.abs()
        saliency = torch.max(grads[0], dim=0)[0].cpu().numpy()
        from scipy.ndimage import gaussian_filter
        saliency = gaussian_filter(saliency, sigma=2)
        saliency_maps.append(saliency)
    
    # Moyenne
    smoothgrad = np.mean(saliency_maps, axis=0)
    
    # Normalisation
    smoothgrad = (smoothgrad - smoothgrad.min()) / (smoothgrad.max() - smoothgrad.min() + 1e-8)
    
    return smoothgrad

def explain_integrated_gradients(model, x, target_idx, device, steps=50, baseline=None):

    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(x)

    # Générer alphas
    alphas = torch.linspace(0, 1, steps).to(device)

    integrated_grads = torch.zeros_like(x)

    for alpha in alphas:
        x_interp = baseline + alpha * (x - baseline)
        x_interp.requires_grad_(True)

        logits = model(x_interp)

        model.zero_grad()
        logits[0, target_idx].backward()

        integrated_grads += x_interp.grad

    # Moyenne
    integrated_grads /= steps

    # Multiplier par (x - baseline)
    integrated_grads = integrated_grads * (x - baseline)

    # Maintenant on retire le batch dimension
    ig = integrated_grads[0].abs()

    # Agrégation RGB
    ig = ig.mean(dim=0)

    # Normalisation
    ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)

    return ig.detach().cpu().numpy()

# ==================== Visualisation ====================

def overlay_heatmap(img01, heatmap, alpha=0.35, colormap='inferno'):
    """
    Overlay heatmap sur image
    """
    # Appliquer colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[..., :3]  # RGB seulement
    
    # Overlay
    overlay = alpha * heatmap_colored + (1 - alpha) * img01
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def overlay_green_heatmap(img01, heatmap, alpha=0.4):

    # Clip valeurs extrêmes (important)
    heatmap = np.clip(heatmap, 0, np.percentile(heatmap, 99))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Créer couche verte
    green_map = np.zeros_like(img01)
    green_map[..., 1] = heatmap  # canal vert uniquement

    overlay = img01.copy()
    overlay = (1 - alpha) * overlay + alpha * green_map
    overlay = np.clip(overlay, 0, 1)

    return overlay

def create_visualization(img01, pil, saliency, smoothgrad, ig, 
                        pred_idx, pred_prob, pred_name, 
                        top_probs, top_labels, model_name, overlay_gradcam):
    """
    Créer visualisation complète
    """
    # Overlays
    overlay_sal = overlay_green_heatmap(img01, saliency)
    overlay_sg = overlay_green_heatmap(img01, smoothgrad)
    overlay_ig = overlay_green_heatmap(img01, ig)
    
    # Figure
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    
    # Image originale
    axes[0, 0].imshow(pil)
    axes[0, 0].set_title("Image Originale (Rétinopathie)", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")
    
    # Top-5 predictions
    topk = len(top_probs)
    y_pos = np.arange(topk)
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(topk)]
    
    axes[0, 1].barh(y_pos, top_probs, color=colors)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(top_labels, fontsize=10)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(0, 1.0)
    axes[0, 1].set_xlabel("Probabilité", fontsize=11)
    axes[0, 1].set_title("Top-5 Prédictions", fontsize=12, fontweight='bold')
    
    for i, p in enumerate(top_probs):
        axes[0, 1].text(p + 0.01, i, f"{p:.3f}", va="center", fontsize=10, fontweight='bold')
    
    # Info modèle
    axes[0, 2].axis("off")
    axes[0, 2].text(0.0, 0.95, "Information Modèle", fontsize=13, fontweight="bold",
                   transform=axes[0, 2].transAxes)
    
    info_text = f"""• Architecture: {model_name}
• Tâche: Classification Rétinopathie
• Prédiction: {pred_name}
  (Classe {pred_idx}, prob={pred_prob:.3f})

Méthodes XAI:
• Gradient-based
• Pixel-level attribution
• 3 méthodes: Saliency, SmoothGrad, IG

Interprétation:
• Régions ROUGES = importantes
• Régions BLEUES = peu importantes
• Focus: vaisseaux, exsudats, hémorragies"""
    
    axes[0, 2].text(0.0, 0.85, info_text, fontsize=10,
                   transform=axes[0, 2].transAxes, va="top",
                   family='monospace')
    
    # Saliency
    axes[1, 0].imshow(overlay_sal)
    axes[1, 0].set_title("Saliency (Vanilla Gradients)", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, -0.05, "Rapide • Peut être bruité", 
                   ha='center', transform=axes[1, 0].transAxes, fontsize=9, style='italic')
    
    # SmoothGrad
    axes[1, 1].imshow(overlay_sg)
    axes[1, 1].set_title("SmoothGrad", fontsize=12, fontweight='bold')
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, -0.05, "Moins de bruit • Plus cohérent", 
                   ha='center', transform=axes[1, 1].transAxes, fontsize=9, style='italic')
    
    # Integrated Gradients
    axes[1, 2].imshow(overlay_ig)
    axes[1, 2].set_title("Integrated Gradients", fontsize=12, fontweight='bold')
    axes[1, 2].axis("off")
    axes[1, 2].text(0.5, -0.05, "Théoriquement fondé • Stable", 
                   ha='center', transform=axes[1, 2].transAxes, fontsize=9, style='italic')
    axes[1, 3].imshow(overlay_gradcam)
    axes[1, 3].set_title("Grad-CAM (ConvNeXt)", fontsize=12, fontweight='bold')
    axes[1, 3].axis("off")
    # Titre global
    fig.suptitle("XAI pour Rétinopathie Diabétique — Modèle Hybride Swin+ConvNeXt", 
                fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94, "Analyse gradient-based : Quelles régions influencent la prédiction ?", 
            ha="center", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    return fig, overlay_sal, overlay_sg, overlay_ig

# ==================== Main ====================
def overlay_cam(img01, cam, alpha=0.4):

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_VIRIDIS  # beaucoup mieux que JET
    )

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0

    overlay = (1 - alpha) * img01 + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay
def deletion_test(model, x, cam, target_idx):
    x_mod = x.clone()

    # supprimer top 10% pixels importants
    threshold = np.percentile(cam, 85)
    cam[cam < threshold] = 0
    mask = torch.tensor(cam > threshold).float().to(x.device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    x_mod = x_mod * (1 - mask)

    with torch.no_grad():
        new_prob = torch.softmax(model(x_mod), dim=1)[0, target_idx]

    return new_prob.item()
def main():
    print("\n" + "="*80)
    print("🔬 XAI POUR RÉTINOPATHIE DIABÉTIQUE — Modèle Hybride Swin+ConvNeXt")
    print("="*80)
    
    # Device
    print(f"\n✅ Device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Charger modèle
    print(f"\n📦 Chargement modèle: {Config.MODEL_PATH}")
    model = ParallelSwinConvNeXt(num_classes=Config.NUM_CLASSES)
    
    try:
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Modèle chargé (epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✅ Modèle chargé")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return
    
    model = model.to(Config.DEVICE)
    model.eval()
    #  Grad-CAM initialisation
    target_layer = model.convnext.stages[-2].blocks[-1]
    gradcam = GradCAM(model, target_layer)
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.41384357213974, 0.22102037072181702, 0.0736616998910904], std=[0.23784145712852478, 0.13096213340759277, 0.04815368354320526])
    ])
    
    # Charger image
    if Config.IMAGE_PATH is None and Config.IMAGE_ID is None:
        # Prendre première image du dataset
        all_images = [f for f in os.listdir(Config.DATA_DIR) if f.endswith('.png')]
        if not all_images:
            print(f"❌ Aucune image trouvée dans {Config.DATA_DIR}")
            return
        Config.IMAGE_ID = all_images[0]
        Config.IMAGE_PATH = os.path.join(Config.DATA_DIR, Config.IMAGE_ID)
    elif Config.IMAGE_PATH is None:
        Config.IMAGE_PATH = os.path.join(Config.DATA_DIR, Config.IMAGE_ID)
    
    print(f"\n🖼️  Image: {os.path.basename(Config.IMAGE_PATH)}")
    
    # Charger et prétraiter
    pil = Image.open(Config.IMAGE_PATH).convert('RGB')
    
    if Config.USE_CONTRAST:
        enhancer = ImageEnhance.Contrast(pil)
        pil = enhancer.enhance(Config.CONTRAST_FACTOR)
    
    # Image pour overlay (0-1)
    img01 = np.asarray(pil.resize((Config.IMG_SIZE, Config.IMG_SIZE))).astype(np.float32) / 255.0
    
    # Tensor pour modèle
    x = transform(pil).unsqueeze(0).to(Config.DEVICE)
    
    # Prédiction
    print("\n🎯 Prédiction...")
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_prob = float(probs[0, pred_idx].item())
        pred_name = Config.CLASS_NAMES[pred_idx]
    cam = gradcam.generate(x, pred_idx)
    overlay_gradcam = overlay_cam(img01, cam)
    mask = np.mean(img01, axis=2) > 0.05
    cam = cam * mask
    print(f"   Prédiction: {pred_name}")
    print(f"   Probabilité: {pred_prob:.2%}")
    
    # Top-5
    topk = 5
    top_probs, top_idx = probs[0].topk(topk)
    top_probs = top_probs.detach().cpu().numpy()
    top_idx = top_idx.detach().cpu().numpy()
    top_labels = [Config.CLASS_NAMES[i] for i in top_idx]
    
    print(f"\n📊 Top-{topk}:")
    for i, (prob, label) in enumerate(zip(top_probs, top_labels)):
        marker = "✅" if i == 0 else "  "
        print(f"   {marker} {label:<20} → {prob:.2%}")
    
    # Appliquer XAI
    print("\n🔬 Application méthodes XAI...")
    
    print("   [1/3] Saliency (Vanilla Gradients)...", end=" ")
    saliency = explain_saliency(model, x, pred_idx, Config.DEVICE)
    print("✅")
    
    print("   [2/3] SmoothGrad...", end=" ")
    smoothgrad = explain_smoothgrad(model, x, pred_idx, Config.DEVICE)
    print("✅")
    
    print("   [3/3] Integrated Gradients...", end=" ")
    ig = explain_integrated_gradients(model, x, pred_idx, Config.DEVICE)
    print("✅")
    
    # Visualisation
    print("\n📊 Création visualisation...")
    model_name = "Swin Tiny + ConvNeXt Tiny"
    
    fig, overlay_sal, overlay_sg, overlay_ig = create_visualization(
        img01, pil, saliency, smoothgrad, ig,
        pred_idx, pred_prob, pred_name,
        top_probs, top_labels, model_name, overlay_gradcam
    )
    
    # Sauvegarder
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(Config.OUTPUT_DIR, "xai_explanation_complete.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Sauvegardé: {output_path}")
    
    # Sauvegarder overlays individuels
    plt.imsave(os.path.join(Config.OUTPUT_DIR, "saliency_overlay.png"), overlay_sal)
    plt.imsave(os.path.join(Config.OUTPUT_DIR, "smoothgrad_overlay.png"), overlay_sg)
    plt.imsave(os.path.join(Config.OUTPUT_DIR, "ig_overlay.png"), overlay_ig)
    plt.imsave(os.path.join(Config.OUTPUT_DIR, "original.png"), img01)
    plt.imsave(os.path.join(Config.OUTPUT_DIR, "GRAD-CAM.png"), overlay_gradcam)
    print(f"   ✅ Overlays sauvegardés dans {Config.OUTPUT_DIR}/")
    
    # Afficher
    plt.show()
    
    print("\n" + "="*80)
    print("🎉 XAI terminé avec succès!")
    print("="*80 + "\n")
    
    print("💡 INTERPRÉTATION:")
    print("   • Régions ROUGES/JAUNES = Pixels qui influencent fortement la prédiction")
    print("   • Régions BLEUES = Pixels qui ont peu d'influence")
    print("   • Focus typique: vaisseaux sanguins, exsudats, hémorragies\n")
    new_prob = deletion_test(model, x, cam, pred_idx)
    print("Probabilité après suppression:", new_prob)
if __name__ == "__main__":
    main()
