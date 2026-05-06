import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# --- PATHS (UNCHANGED) ---
image_train = '/users/mnarayan/data/mnarayan/mini_imagenet/train'
image_val = '/users/mnarayan/data/mnarayan/mini_imagenet/val'
image_test = '/users/mnarayan/data/mnarayan/mini_imagenet/test'

size = 84
num_train_augs = 10

def list_files(filepath, filetype):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith(filetype):
                paths.append(os.path.join(root, file))
    return paths

# --- TRANSFORMS ---
# 1. Base resize for the "original" training image (aug_00)
transform_train_base = transforms.Compose([
    transforms.Resize((size, size)), # Direct squish to 84x84 for perfect topology
    transforms.ToTensor()
])

# 2. Random augmentations for the remaining 9 training images
transform_train_random = transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# 3. Deterministic eval transform for val/test
transform_eval = transforms.Compose([
    transforms.Resize([92, 92]),
    transforms.CenterCrop(size),
    transforms.ToTensor()
])

# --- PROCESSING FUNCTIONS ---
def process_train_images(paths):
    print(f"Processing {len(paths)} training images...")
    for p in paths:
        img = Image.open(p).convert('RGB')
        
        for i in range(num_train_augs):
            # Formats index with leading zero (e.g., _aug_00.jpg, _aug_01.jpg)
            aug_fname = p.replace('.jpg', f'_aug_{i:02d}.jpg')
            
            # aug_00 is the safe direct resize; the rest are random crops
            if i == 0:
                img_tensor = transform_train_base(img)
            else:
                img_tensor = transform_train_random(img)
                
            save_image(img_tensor, aug_fname)

def process_eval_images(paths, split_name):
    print(f"Processing {len(paths)} {split_name} images...")
    for p in paths:
        img = Image.open(p).convert('RGB')
        
        # Following your logic to name the single eval image _aug_00.jpg
        eval_fname = p.replace('.jpg', '_aug_00.jpg') 
        img_eval = transform_eval(img)
        save_image(img_eval, eval_fname)

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Gather files
    paths_train = list_files(image_train, '.jpg')
    paths_val = list_files(image_val, '.jpg')
    paths_test = list_files(image_test, '.jpg')

    # Execute processing
    process_train_images(paths_train)
    process_eval_images(paths_val, 'validation')
    process_eval_images(paths_test, 'test')
    
    print("Done generating offline augmentations!")
