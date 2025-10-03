from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from typing import List, Dict, Any, Tuple

from utils import search_animal_class, sorted_classes


# --- GLOBAL MODEL INITIALIZATION (LOAD ONCE) ---
BASE_MODEL = "jayant-yadav/roberta-base-multinerd"

TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)
MODEL = AutoModelForTokenClassification.from_pretrained(BASE_MODEL)

NLP_PIPELINE = pipeline("ner", model=MODEL, tokenizer=TOKENIZER, aggregation_strategy="simple")
# ---------------------------------------------


def inference(text: str) -> List[Dict[str, Any]]:
    """
    Performs NER and processes 'B-ANIM' entities.
    """
    ner_results = NLP_PIPELINE(text)
    
    print(f"--- Input: {text} ---")
    
    for ner_item in ner_results:
        entity_tag = ner_item.get("entity_group", ner_item.get("entity")) 
        print(ner_item) # Essential output of NER result

        if entity_tag in ["B-ANIM", "ANIM"]:
            query = ner_item["word"].strip().lstrip('Ġ').replace('##', '')
            
            matches = search_animal_class(query, sorted_classes, threshold=0.5)

            print(f"  🔍 Searching '{query}':")
            if matches:
                for name, dist in matches:
                      print(f"    -> MATCH: {name} (Dist: {dist:.4f})")
            else:
                print(f"    -> No match found for '{query}'.")
        
    return ner_results


if __name__ == "__main__":
    
    # Example 1: Human and Location (should not trigger animal search)
    example1 = "My name is Wolfgang and I live in Berlin."
    inference(example1)
    
    # Example 2: Animal (should trigger animal search)
    example2 = "A quick brown fox jumped over the lazy dog."
    inference(example2)