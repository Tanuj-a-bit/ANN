import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from src.model import HandwritingModel
from src.dataset import HandwritingDataset, collate_fn
from src.utils import decode_prediction, calculate_metrics

def load_manifest(manifest_path):
    image_paths = []
    labels = []
    dir_path = os.path.dirname(manifest_path)
    split = "train" if "train" in manifest_path else "val"
    
    with open(manifest_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                image_paths.append(os.path.join(DATA_DIR, split, parts[0]))
                labels.append(parts[1])
    return image_paths, labels

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load data
    train_manifest = os.path.join(DATA_DIR, "train_manifest.txt")
    val_manifest = os.path.join(DATA_DIR, "val_manifest.txt")
    
    if not os.path.exists(train_manifest):
        print("Data manifests not found. Please run scripts/generate_data.py first.")
        return

    train_imgs, train_labels = load_manifest(train_manifest)
    val_imgs, val_labels = load_manifest(val_manifest)
    
    train_dataset = HandwritingDataset(train_imgs, train_labels)
    val_dataset = HandwritingDataset(val_imgs, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model, Loss, Optimizer
    model = HandwritingModel(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            images, labels, input_lengths, label_lengths, _ = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images) # (seq_len, batch_size, num_classes)
            
            # CTC Loss
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        val_loss = 0
        val_cer = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels, input_lengths, label_lengths, texts = batch
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels, input_lengths, label_lengths)
                val_loss += loss.item()
                
                preds = decode_prediction(outputs)
                val_cer += calculate_metrics(preds, texts)
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_cer = val_cer / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val CER: {avg_cer:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("Checkpoint saved!")

if __name__ == "__main__":
    train()
