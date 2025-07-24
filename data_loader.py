import os
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoProcessor
from config import IMAGE_DIR, DESCRIPTION_CSV, BATCH_SIZE, MODEL_NAME, MAX_TEXT_LENGTH, VALIDATION_SPLIT, RANDOM_SEED

def validate_image_path(image_path):
    """Check if image exists and is readable"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, UnidentifiedImageError, OSError):
        return False

class FashionDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor
        self.valid_indices = self._prevalidate_images()

    def _prevalidate_images(self):
        """Pre-check all images during initialization"""
        valid = []
        image_dir = Path(IMAGE_DIR)
        for idx in tqdm(range(len(self.dataframe)), desc="Prevalidating images"):
            product_id = self.dataframe.iloc[idx]['product_id']
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = image_dir / f"{product_id}{ext}"
                if img_path.exists() and validate_image_path(img_path):
                    valid.append(idx)
                    break
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.dataframe.iloc[real_idx]

        image_path = Path(IMAGE_DIR) / row['image_path']  # Use stored valid path
        text = row['description'][:MAX_TEXT_LENGTH]

        try:
            image = Image.open(image_path).convert("RGB")

            # Process inputs
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding='max_length',
                max_length=MAX_TEXT_LENGTH,
                truncation=True
            )

            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'input_ids': inputs['input_ids'].squeeze(0),
                'product_id': row['product_id']
            }
        except Exception as e:
            print(f"Error loading {image_path}: {str(e)}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'product_ids': [b['product_id'] for b in batch]
    }

def create_data_loaders():
    print("Building validated dataset...")
    df = pd.read_csv(DESCRIPTION_CSV)
    df['product_id'] = df['product_id'].astype(str)

    valid_entries = []
    image_dir = Path(IMAGE_DIR)
    for product_id in tqdm(df['product_id'], desc="Finding valid images"):
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = image_dir / f"{product_id}{ext}"
            if img_path.exists() and validate_image_path(img_path):
                valid_entries.append({
                    'product_id': product_id,
                    'description': df[df['product_id'] == product_id]['description'].values[0],
                    'image_path': str(img_path.relative_to(IMAGE_DIR)) # Store relative path
                })
                break
    df_valid = pd.DataFrame(valid_entries)
    print(f"Valid dataset contains {len(df_valid)} entries")

    processor = AutoProcessor.from_pretrained(MODEL_NAME,fast=True)
    dataset = FashionDataset(df_valid, processor)

    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(), # Try using multiple workers
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(), # Try using multiple workers
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, processor