import torch

# Configuration for Handwriting Recognition System

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_LSTM_LAYERS = 2
HIDDEN_SIZE = 256

# Character Set
# Using lowercase letters, uppercase letters, digits, and some punctuation
VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
CHAR_TO_IDX = {char: i + 1 for i, char in enumerate(VOCAB)}
IDX_TO_CHAR = {i + 1: char for i, char in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB) + 1  # +1 for CTC blank token

# Paths
DATA_DIR = "/Users/tanujs/Desktop/ANN/data"
CHECKPOINT_DIR = "/Users/tanujs/Desktop/ANN/checkpoints"
LOG_DIR = "/Users/tanujs/Desktop/ANN/logs"

# Preprocessing
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
