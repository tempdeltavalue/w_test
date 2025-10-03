# !pip install transformers datasets accelerate -U
# !pip install torch tqdm scikit-learn pandas
import torch
import numpy as np
import sys
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
import json
import pandas as pd
from typing import List, Tuple, Dict, Any

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================================
# 1. MODEL & LABEL SETUP
# ======================================================================

MODEL_NAME = "jayant-yadav/roberta-base-multinerd"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model_config = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).config

BASE_MODEL_LABELS: List[str] = list(base_model_config.id2label.values())
num_full_labels = len(BASE_MODEL_LABELS)

# Target label setup: We map the new "B-ANIMAL" tag to the existing "B-ANIM" ID
TARGET_LABEL_IN_DATA = "B-ANIMAL"
TARGET_LABEL_IN_MODEL = "B-ANIM"

try:
    ID_O = base_model_config.label2id["O"]
    ID_B_ANIM = base_model_config.label2id[TARGET_LABEL_IN_MODEL]
except KeyError:
    print(f"❌ Error: '{TARGET_LABEL_IN_MODEL}' or 'O' not found in model's labels. Check model documentation.")
    sys.exit(1)

label_to_id: Dict[str, int] = base_model_config.label2id
id_to_label: Dict[int, str] = base_model_config.id2label

# Metric functions (using TARGET_LABEL_IN_MODEL, which is 'B-ANIM')
def get_entities(labels: List[str]) -> set:
    entities = []
    for i, label in enumerate(labels):
        if label == TARGET_LABEL_IN_MODEL: 
            entities.append(('ANIM', i, i + 1))
    return set(entities)

def custom_ner_metrics(true_labels_list: List[List[str]], pred_labels_list: List[List[str]]) -> Dict[str, float]:
    total_true, total_pred, correct_pred = 0, 0, 0
    
    def filter_labels(labels: List[str]) -> List[str]:
        return [l if l == TARGET_LABEL_IN_MODEL else 'O' for l in labels]

    for true_labels, pred_labels in zip(true_labels_list, pred_labels_list):
        true_entities = get_entities(filter_labels(true_labels))
        pred_entities = get_entities(filter_labels(pred_labels))
        
        correct_pred += len(true_entities.intersection(pred_entities))
        total_true += len(true_entities)
        total_pred += len(pred_entities)

    precision = correct_pred / total_pred if total_pred > 0 else 0
    recall = correct_pred / total_true if total_true > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall}


# ----------------------------------------------------------------------
# 2. DATA LOADING (MODIFIED FOR FOLDERS)
# ----------------------------------------------------------------------

def load_data_from_folders(dataset_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Loads tokens and labels from all labels.json files in subfolders."""
    all_tokens: List[List[str]] = []
    all_labels: List[List[str]] = []
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        sys.exit(1)

    # Iterate over all items in the root path
    for item_name in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item_name)
        
        # Check if it's a subfolder
        if os.path.isdir(item_path):
            json_file_path = os.path.join(item_path, "labels.json")
            
            if os.path.exists(json_file_path):
                print(f"Loading data from: {item_name}")
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Ensure data has required keys
                    if "sentences" in data and "labels" in data:
                        all_tokens.extend(data["sentences"])
                        
                        # --- REMAPPING LABELS ---
                        # Map 'B-ANIMAL' to 'B-ANIM' for model compatibility
                        remapped_labels = []
                        for sentence_labels in data["labels"]:
                            remapped_labels.append([
                                TARGET_LABEL_IN_MODEL if label == TARGET_LABEL_IN_DATA else label
                                for label in sentence_labels
                            ])
                        all_labels.extend(remapped_labels)
                    else:
                        print(f"⚠️ Warning: {json_file_path} is missing 'sentences' or 'labels' key.")
            else:
                print(f"⚠️ Warning: {json_file_path} not found.")

    if not all_tokens:
        print("❌ Error: No data loaded. Check file structure and paths.")
        sys.exit(1)
        
    print(f"\n✅ Successfully loaded {len(all_tokens)} total sentences.")
    return all_tokens, all_labels

# --- Load the actual dataset ---
dataset_path = "ner_data_tinyllama_batch" 
# NOTE: Replace 'ner_data_tinyllama_batch' with the actual path if running outside this environment
raw_tokens, raw_labels = load_data_from_folders(dataset_path)

# Convert labels to IDs
raw_labels_id = [[label_to_id[l] for l in tags] for tags in raw_labels]
data_dict = {"id": list(range(len(raw_tokens))), "tokens": raw_tokens, "ner_tags": raw_labels_id}
raw_datasets = Dataset.from_dict(data_dict)

# Split dataset
TEST_SIZE = 0.2
dataset_split = raw_datasets.train_test_split(test_size=TEST_SIZE, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")


# ----------------------------------------------------------------------
# 3. TOKENIZATION AND DATA LOADERS
# ----------------------------------------------------------------------

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="longest")
    labels = []

    for i in range(len(examples["ner_tags"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        current_labels = examples["ner_tags"][i]
        previous_word_idx = None
        label_ids = []
        max_label_index = len(current_labels) - 1

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx == previous_word_idx:
                label_ids.append(ID_O) 
            elif word_idx > max_label_index:
                label_ids.append(-100)
            else:
                label_ids.append(current_labels[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

cols_to_keep = ['input_ids', 'attention_mask', 'labels']
tokenized_train_dataset.set_format(type="torch", columns=cols_to_keep)
tokenized_eval_dataset.set_format(type="torch", columns=cols_to_keep)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# You may need to adjust batch size based on your GPU memory
TRAIN_BATCH_SIZE = 8 
EVAL_BATCH_SIZE = 16

train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=EVAL_BATCH_SIZE, collate_fn=data_collator)

## --- Weighted Loss Calculation ---
num_o = sum(row.count(ID_O) for row in raw_labels_id)
num_animal = sum(row.count(ID_B_ANIM) for row in raw_labels_id)

weight_o = 1.0
weight_animal = num_o / num_animal # Higher weight for the minority class

weights = torch.ones(num_full_labels) 
weights[ID_O] = weight_o
weights[ID_B_ANIM] = weight_animal
weights = weights.to(device)
print(f"Class Weights (O:{weight_o:.2f}, {TARGET_LABEL_IN_MODEL}:{weight_animal:.2f}) applied.")


# ----------------------------------------------------------------------
# 4. MODEL AND TRAINING LOOP SETUP (EARLY STOPPING & HISTORY)
# ----------------------------------------------------------------------

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
model.to(device)

# --- Layer Freezing Logic ---
# Freeze all parameters of the RoBERTa body (Transformer Encoder)
for name, param in model.roberta.named_parameters():
    param.requires_grad = False

# Unfreeze only the classification head and the last encoder layer
for param in model.classifier.parameters():
    param.requires_grad = True
try:
    for param in model.roberta.encoder.layer[-1].parameters():
        param.requires_grad = True
except AttributeError:
    pass 
    
optimizer = AdamW(model.parameters(), lr=5e-5) 
num_epochs = 50
loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

# --- Early Stopping and History Configuration ---
PATIENCE = 5 
best_f1 = -1.0
epochs_no_improve = 0
best_model_state = None
MODEL_SAVE_PATH = "finetuned_multinerd_animal_batch"
training_history: List[Dict[str, Any]] = []


# --- Helper function to test model on a fixed sentence ---
def test_model_on_sentence(model, tokenizer, device, sentence):
    model.eval()
    tokenized_input = tokenizer.encode_plus(sentence, return_tensors="pt", is_split_into_words=False)
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    predicted_tags = [id_to_label[p] for p in predictions] 
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    
    print("\n   >>> Prediction on test sentence (Highlighted Tokens):")
    decoded_output = []
    
    for token, tag in zip(decoded_tokens, predicted_tags):
        clean_token = token.strip().replace('Ġ', '').replace('#', '')
        
        if tag.endswith('ANIM'): 
            decoded_output.append(f"**{clean_token}** [{tag}]")
        elif tag == 'O' or tag in ['<s>', '</s>', '<pad>']:
            decoded_output.append(f"{token}")
        else:
            decoded_output.append(f"{clean_token} [{tag}]") 

    print("   ", " ".join(decoded_output))
    model.train()

# --- Function to save history to files ---
def save_history_to_file(history: List[Dict[str, Any]], path: str):
    """Saves the training history to both a human-readable text file and a JSON file."""
    if not history:
        print("\n⚠️ Warning: Training history is empty, skipping save.")
        return

    try:
        df = pd.DataFrame(history)
        
        # 1. Save to TXT file
        file_path = os.path.join(path, "training_history.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("--- Training History (B-ANIM Fine-tuning) ---\n\n")
            f.write(df.to_string(index=False, float_format="%.6f"))
            f.write(f"\n\nBest Val F1: {df['Val F1'].max():.4f}")
            f.write(f"\nTotal Epochs Run: {len(history)}")

        # 2. Save to JSON file
        json_path = os.path.join(path, "training_history.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
            
        print(f"\n✅ History saved to: {file_path} and {json_path}")
    except Exception as e:
        print(f"\n❌ Error saving history file: {e}")


print("\n🚀 Starting PyTorch Training Loop with Early Stopping & History...")
# Test sentence to check original NER capabilities and new 'fox' recognition
test_sentence = "The CEO of Google visited Berlin and saw a small fox."

# --- Training Loop ---
for epoch in range(num_epochs):

    # === 1. TRAINING PHASE ===
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        # Use custom weighted loss
        loss = loss_fn(logits.view(-1, num_full_labels), batch["labels"].view(-1)) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)

    # === 2. EVALUATION PHASE ===
    model.eval()
    all_true_labels, all_pred_labels = [], []
    val_loss = 0.0

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            
            loss = loss_fn(logits.view(-1, num_full_labels), batch["labels"].view(-1))
            val_loss += loss.item() * batch['input_ids'].size(0)

            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for label_seq, pred_seq in zip(labels, predictions):
                true_labels = [id_to_label[l] for l in label_seq if l != -100]
                pred_labels = [id_to_label[p] for (p, l) in zip(pred_seq, label_seq) if l != -100]
                if len(true_labels) == len(pred_labels):
                    all_true_labels.append(true_labels)
                    all_pred_labels.append(pred_labels)
                    
    avg_val_loss = val_loss / len(eval_dataloader.dataset)
    metrics = custom_ner_metrics(all_true_labels, all_pred_labels) 
    current_f1 = metrics['f1']

    # === 3. RECORD HISTORY ===
    history_entry = {
        "Epoch": epoch + 1,
        "Train Loss": avg_train_loss,
        "Val Loss": avg_val_loss,
        "Val F1": current_f1,
        "Val Precision": metrics['precision'],
        "Val Recall": metrics['recall']
    }
    training_history.append(history_entry)

    # === 4. EARLY STOPPING AND SAVE LOGIC ===
    is_best = current_f1 > best_f1
    if is_best:
        best_f1 = current_f1
        epochs_no_improve = 0
        best_model_state = model.state_dict()
        print(f"⭐️ New best F1: {best_f1:.4f}. Model state saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs.")

    # --- PRINT RESULTS and TEST ---
    print(f"\n| Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
          f"Val F1 ({TARGET_LABEL_IN_MODEL}): {current_f1:.4f} | Val P: {metrics['precision']:.4f} | Val R: {metrics['recall']:.4f} |")
    print("-" * 80)

    test_model_on_sentence(model, tokenizer, device, test_sentence)
    print("=" * 80)
    
    # Early stopping condition
    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping triggered after {epoch+1} epochs due to no F1 improvement for {PATIENCE} epochs.")
        break


# ----------------------------------------------------------------------
# 5. FINAL SAVE AND HISTORY
# ----------------------------------------------------------------------

if best_model_state is not None:
    print(f"\n💾 Loading best model state (F1: {best_f1:.4f}) and saving to disk at: {MODEL_SAVE_PATH}...")
    
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs