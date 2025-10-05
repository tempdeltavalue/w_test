import torch
import os
import sys
import argparse
from PIL import Image
from typing import List, Dict, Any

from image_clf.models import get_mobilenet_v2, mobilenet_v2_transform 
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ner.utils import search_animal_class, sorted_classes

# --- CONFIGURATION ---
DEFAULT_IMG_MODEL_PATH = "training_output_mobilenet/best_mobilenetv2_finetuned.pth"
DEFAULT_NUM_CLASSES = 99 
BASE_MODEL = "jayant-yadav/roberta-base-multinerd"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# 1. GLOBAL MODEL INITIALIZATION (LOAD ONCE)
# ======================================================================

try:
    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)
    MODEL = AutoModelForTokenClassification.from_pretrained(BASE_MODEL)
    NLP_PIPELINE = pipeline("ner", model=MODEL, tokenizer=TOKENIZER, aggregation_strategy="simple")
except Exception as e:
    print(f"❌ Error loading NER model: {e}")
    sys.exit(1)

IMAGE_CLASSIFIER = None

# ======================================================================
# 2. CORE FUNCTIONS
# ======================================================================

def load_image_model(model_path: str, num_classes: int):
    if not os.path.exists(model_path):
        print(f"❌ Weights not found at: {model_path}")
        return None
    
    img_model = get_mobilenet_v2(num_classes, DEVICE, pretrained=False).eval()
    
    try:
        img_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return img_model
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None

def classify_local_image(image_path: str, model: torch.nn.Module, class_names: List[str]):
    if not os.path.exists(image_path):
        return {"result": "NOT FOUND", "error": f"Image not found at {image_path}"}

    transform = mobilenet_v2_transform(is_train=False) 
    image = Image.open(image_path).convert('RGB')
    input_batch = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_p, top_class = probabilities.topk(1, dim=0)
    
    return {
        "result": "OK",
        "top_class_name": class_names[top_class[0].item()],
        "confidence": top_p[0].item(),
    }


def ner_and_classify(text: str, image_dir: str, class_names: List[str]):
    
    ner_results = NLP_PIPELINE(text)
    
    print(f"\n--- Text: '{text[:60]}...' ({DEVICE}) ---")
    
    for ner_item in ner_results:
        entity_tag = ner_item.get("entity_group", ner_item.get("entity")) 

        if entity_tag in ["B-ANIM", "ANIM"]:
            query = ner_item["word"].strip().lstrip('Ġ').replace('##', '')
            
            matches = search_animal_class(query, sorted_classes, threshold=0.5)
            
            print(f"\n[NER] Animal found: '{query}'")
            
            if matches:
                matched_class, _ = matches[0]
                print(f"[SEARCH] Matched class: {matched_class}")
                
                # Simulate image path based on class name
                image_path = os.path.join(image_dir, matched_class, f"sample_{matched_class}.jpg")
                
                if IMAGE_CLASSIFIER:
                    classification_result = classify_local_image(image_path, IMAGE_CLASSIFIER, class_names)
                    
                    if classification_result['result'] == 'OK':
                        print(f"[IMG] Classified as: {classification_result['top_class_name']} ({classification_result['confidence']:.4f})")
                    else:
                         print(f"[IMG] Classification failed: {classification_result['error']}")
            
            else:
                print(f"[SEARCH] No matching class found for '{query}'.")
                
    if not any(item.get("entity_group") in ["B-ANIM", "ANIM"] for item in ner_results):
        print("No animal entities found.")


# ======================================================================
# 3. MAIN
# ======================================================================

def main():
    global IMAGE_CLASSIFIER
    parser = argparse.ArgumentParser(description="Combined Text (NER) and Image Classification Pipeline.")
    
    parser.add_argument('text', type=str, help="Text to process for animal entities.")
    parser.add_argument('--image_model_path', type=str, default=DEFAULT_IMG_MODEL_PATH)
    parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument('--image_dir', type=str, default="datasets/animals_photo_sets/animals99_subset",
                        help="Root directory where the classification image folders are stored (for simulation).")
    
    args = parser.parse_args()

    class_names = sorted_classes 
    
    IMAGE_CLASSIFIER = load_image_model(args.image_model_path, args.num_classes)
    if not IMAGE_CLASSIFIER:
        sys.exit(1)

    ner_and_classify(args.text, args.image_dir, class_names)
    

if __name__ == "__main__":
    main()