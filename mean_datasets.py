import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DATA_DIR = "C:/Users/eljat/Desktop/Projet_GetHub/fundus-squaring/train_images"
IMG_SIZE = 224  # même taille que ton training

# Transform SANS normalisation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==============================
# CALCUL
# ==============================
mean = torch.zeros(3)
std = torch.zeros(3)
n_images = 0

image_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".png")]

for img_name in tqdm(image_files):
    img_path = os.path.join(DATA_DIR, img_name)
    
    img = Image.open(img_path).convert("RGB")
    img = transform(img)  # shape: (3, H, W)
    
    mean += img.mean(dim=[1,2])
    std += img.std(dim=[1,2])
    n_images += 1

mean /= n_images
std /= n_images

print("\n📊 Dataset Statistics:")
print("Mean:", mean.tolist())
print("Std:", std.tolist())

'''
    📊 Dataset Statistics:
        Mean: [0.41384357213974, 0.22102037072181702, 0.0736616998910904]
    Std: [0.23784145712852478, 0.13096213340759277, 0.04815368354320526]
    '''