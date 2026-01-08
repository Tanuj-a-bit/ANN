import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import CHAR_TO_IDX, IMG_HEIGHT, IMG_WIDTH

class HandwritingDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def preprocess_image(self, img_path):
        # Load as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Create a blank image if path is invalid
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        
        # Resize to fixed height while maintaining aspect ratio or pad
        h, w = img.shape
        new_w = int(h * (IMG_WIDTH / IMG_HEIGHT))
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Ensure ink is high value (1.0) and paper is low value (0.0)
        if np.mean(img) > 0.5:
            img = 1.0 - img
            
        img = np.expand_dims(img, axis=0) # Add channel dim
        return img

    def encode_label(self, label):
        return [CHAR_TO_IDX[char] for char in label if char in CHAR_TO_IDX]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.preprocess_image(img_path)
        
        label = self.labels[idx]
        encoded_label = self.encode_label(label)
        
        return {
            'image': torch.from_numpy(image),
            'label': torch.LongTensor(encoded_label),
            'label_length': torch.IntTensor([len(encoded_label)]),
            'text': label
        }

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    label_lengths = torch.cat([item['label_length'] for item in batch])
    texts = [item['text'] for item in batch]
    
    # Sequence length after CNN (calculated from model architecture)
    # Our model reduces width from 128 to 32
    input_lengths = torch.full(size=(len(batch),), fill_value=32, dtype=torch.int32)
    
    return images, labels, input_lengths, label_lengths, texts
