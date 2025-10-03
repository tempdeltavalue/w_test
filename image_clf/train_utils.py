import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import copy
from typing import Dict, List, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(DATASET_PATH, BATCH_SIZE, TEST_SPLIT_RATIO, data_transforms):
    full_dataset = datasets.ImageFolder(DATASET_PATH, data_transforms['val'])
    NUM_CLASSES = len(full_dataset.classes)
    print(f"Loaded {len(full_dataset)} images across {NUM_CLASSES} classes.")

    # Calculate split sizes
    test_size = int(TEST_SPLIT_RATIO * len(full_dataset))
    train_size = len(full_dataset) - test_size

    # Split indices randomly
    train_data_indices, test_data_indices = torch.utils.data.random_split(
        full_dataset,
        [train_size, test_size]
    )

    # Apply correct transforms and create subsets
    train_dataset = datasets.ImageFolder(DATASET_PATH, data_transforms['train'])
    test_dataset = datasets.ImageFolder(DATASET_PATH, data_transforms['val'])

    train_subset = torch.utils.data.Subset(train_dataset, train_data_indices.indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_data_indices.indices)

    # Create DataLoaders (num_workers=0 to fix potential Colab/Jupyter issues)
    dataloaders = {
        'train': DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }


    return dataloaders, full_dataset.classes

def save_and_print_history(history: Dict, filename: str):
    """Saves training history to a file and prints a summary."""
    print("\n--- Training History Summary ---")

    with open(filename, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Train_Acc,Val_Acc\n")

        for i in range(len(history['train_loss'])):
            t_loss = history['train_loss'][i]
            v_loss = history['val_loss'][i]
            v_acc = history['val_acc'][i]

            f.write(f"{i+1},{t_loss:.4f},{v_loss:.4f},{history['train_acc'][i]:.4f},{v_acc:.4f}\n")

            print(f"E{i+1} | T_Loss: {t_loss:.4f} | V_Loss: {v_loss:.4f} | V_Acc: {v_acc:.4f}")

    print(f"\n✅ Training history saved to {filename}")


def evaluate_model(model, dataloader, criterion, class_names: List[str]) -> Tuple[float, float, str, List[int], List[int]]:
    """Evaluates the model and returns loss, accuracy, and classification report."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    # FIX: Pass 'labels' explicitly (0 to NUM_CLASSES - 1) to handle missing classes in the test split
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    return total_loss, total_acc.item(), report, all_labels, all_preds

def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_save_path, report_file_best, class_names):
    print("\n--- Starting Training ---")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(dataloaders['train'].dataset)
        train_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())

        # --- VALIDATION PHASE ---
        val_loss, val_acc, val_report, _, _ = evaluate_model(
            model,
            dataloaders['val'],
            criterion,
            class_names
        )
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss:   {val_loss:.4f} Acc: {val_acc:.4f}')

        # --- SAVE BEST MODEL LOGIC ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save weights
            torch.save(model.state_dict(), model_save_path)

            # Save classification report for the best model
            with open(report_file_best, 'w') as f:
                f.write(f"Best Model (Epoch {epoch+1}) | Validation Loss: {best_val_loss:.4f}\n")
                f.write("-" * 35 + "\n")
                f.write(val_report)

            print("\n--- CLASSIFICATION REPORT (BEST EPOCH) ---")
            print(val_report)
            print("------------------------------------------")
            print(f"*** NEW BEST MODEL SAVED! Val Loss: {best_val_loss:.4f} ***")

    # Load the best model weights found during training before returning
    model.load_state_dict(best_model_wts)
    return model, history