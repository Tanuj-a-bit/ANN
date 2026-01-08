import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import DATA_DIR, VOCAB, IMG_HEIGHT, IMG_WIDTH

def generate_synthetic_data(num_samples=1000, split='train'):
    output_dir = os.path.join(DATA_DIR, split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find a handwriting-like font, fallback to default
    font_paths = [
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/System/Library/Fonts/MarkerFelt.ttc",
        "/System/Library/Fonts/Noteworthy.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf"
    ]
    
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 24)
                break
            except:
                continue
    
    if font is None:
        font = ImageFont.load_default()

    manifest = []
    
    print(f"Generating {num_samples} samples for {split}...")
    for i in range(num_samples):
        # Generate random text
        length = random.randint(3, 10)
        text = "".join(random.choices(VOCAB.strip(), k=length))
        
        # Create image (Black background)
        img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), color=0)
        draw = ImageDraw.Draw(img)
        
        # ... get text size ...
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        x = random.randint(0, max(0, IMG_WIDTH - text_w))
        y = random.randint(0, max(0, IMG_HEIGHT - text_h))
        
        # White text
        draw.text((x, y), text, fill=255, font=font)
        
        # Add some "handwriting" noise/distortion
        img_np = np.array(img)
        
        # Slight elastic transformation or blur
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
        
        # Save image
        img_filename = f"{i}.png"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, img_np)
        
        manifest.append(f"{img_filename}\t{text}")
        
    with open(os.path.join(DATA_DIR, f"{split}_manifest.txt"), "w") as f:
        f.write("\n".join(manifest))

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_synthetic_data(2000, 'train')
    generate_synthetic_data(400, 'val')
