import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import sys
import argparse
from typing import Dict, List, Tuple

from models import get_mobilenet_v2, mobilenet_v2_transform
from train_utils import get_dataloaders, train_model, save_and_print_history, DEVICE, PATIENCE_DEFAULT # Assume PATIENCE_DEFAULT is defined here

# --- CONFIGURATION (Default Values) ---
# We define defaults here, but they will be overwritten by command-line args
DEFAULT_DATASET_PATH = "datasets/animals_photo_sets/animals99_subset"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 50 
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TEST_SPLIT_RATIO = 0.2
DEFAULT_OUTPUT_DIR = "training_output_mobilenet"
# --------------------------------------

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for animal classification with Early Stopping.")
    
    # Data and Path Arguments
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATASET_PATH,
                        help=f"Path to the root dataset directory. Default: {DEFAULT_DATASET_PATH}")
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the model, reports, and history. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument('--test_split', type=float, default=DEFAULT_TEST_SPLIT_RATIO,
                        help=f"Ratio of the dataset to use for validation/test split. Default: {DEFAULT_TEST_SPLIT_RATIO}")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS,
                        help=f"Maximum number of epochs to train. Default: {DEFAULT_NUM_EPOCHS}")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training and validation. Default: {DEFAULT_BATCH_SIZE}")
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Learning rate for the Adam optimizer. Default: {DEFAULT_LEARNING_RATE}")
    
    # Early Stopping
    parser.add_argument('--patience', type=int, default=PATIENCE_DEFAULT, # Assuming this PATIENCE_DEFAULT is imported from train_utils
                        help=f"Number of epochs without validation loss improvement before early stopping. Default: {PATIENCE_DEFAULT}")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- SETUP FILE PATHS ---
    output_dir = args.output_dir
    MODEL_SAVE_PATH = os.path.join(output_dir, "best_mobilenetv2_finetuned.pth")
    REPORT_FILE_BEST = os.path.join(output_dir, "best_classification_report.txt")
    HISTORY_FILE = os.path.join(output_dir, "training_history.txt")

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"--- Training Configuration ---")
    print(f"Dataset Path: {args.data_path}")
    print(f"Max Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Early Stopping Patience: {args.patience}")
    print(f"Using device: {DEVICE}")
    print("-" * 30)

    # Ensure deterministic behavior for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 1. DATA LOADING ---
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Dataset path not found: {args.data_path}")
        sys.exit(1)

    # Get transformations
    data_transforms = mobilenet_v2_transform()

    # Get data loaders
    dataloaders, class_names = get_dataloaders(args.data_path, args.batch_size, args.test_split, data_transforms)
    NUM_CLASSES = len(class_names)
    print(f"Loaded {len(dataloaders['train'].dataset) + len(dataloaders['val'].dataset)} images across {NUM_CLASSES} classes.")
    # print(f"Class names (sorted): {class_names}")

    # --- 2. MODEL, CRITERION, OPTIMIZER SETUP ---
    # get_mobilenet_v2 is assumed to handle freezing and head modification
    model_ft = get_mobilenet_v2(NUM_CLASSES, DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Pass args.lr (Learning Rate) to the optimizer
    optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)

    # --- 3. TRAINING ---
    model_ft, history = train_model(
        model_ft,
        dataloaders,
        criterion,
        optimizer,
        args.epochs,        # Max epochs
        MODEL_SAVE_PATH,
        REPORT_FILE_BEST,
        class_names,
        args.patience         # Early Stopping Patience
    )

    # --- 4. FINAL REPORTING ---
    save_and_print_history(history, HISTORY_FILE)
    
    print("\n✅ Training script finished.")


if __name__ == "__main__":
    main()