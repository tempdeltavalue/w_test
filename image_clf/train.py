import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import sys


from models import get_mobilenet_v2, mobilenet_v2_transform
from train_utils import get_dataloaders, train_model, save_and_print_history, DEVICE

# --- CONFIGURATION ---
DATASET_PATH = "datasets/animals_photo_sets/animals99_subset"
BATCH_SIZE = 32
NUM_EPOCHS = 10 # Set to a higher number for real training
LEARNING_RATE = 0.001
TEST_SPLIT_RATIO = 0.2
MODEL_SAVE_PATH = "best_mobilenetv2_finetuned.pth"
REPORT_FILE_BEST = "best_classification_report.txt"
HISTORY_FILE = "training_history.txt"
print(f"Using device: {DEVICE}")


# Ensure deterministic behavior for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset path not found: {DATASET_PATH}")
    sys.exit(1)

# Data transformations (standard for MobileNetV2)
data_transforms = mobilenet_v2_transform()

dataloaders, class_names = get_dataloaders(DATASET_PATH, BATCH_SIZE, TEST_SPLIT_RATIO, data_transforms)
NUM_CLASSES = len(class_names)

print(f"Class names (sorted): {class_names}")

model_ft = get_mobilenet_v2(NUM_CLASSES, DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

model_ft, history = train_model(
    model_ft,
    dataloaders,
    criterion,
    optimizer,
    NUM_EPOCHS,
    MODEL_SAVE_PATH,
    REPORT_FILE_BEST,
    class_names
)


save_and_print_history(history, HISTORY_FILE)