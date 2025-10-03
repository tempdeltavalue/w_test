import os
import sys
from typing import List, Tuple

# --- JACCARD DISTANCE UTILITY ---

CLASSES_FILE = "datasets/ner_sets/all_animals_names.txt"

def _get_ngram_set(text: str, n: int = 2) -> set:
    """Converts text into a set of bigrams for similarity calculation."""
    text = " " + text.lower().strip() + " "
    return set(text[i:i + n] for i in range(len(text) - n + 1))

def jaccard_distance(s1: str, s2: str, n: int = 2) -> float:
    """Calculates Jaccard Distance (1 - Jaccard Similarity) based on n-grams."""
    set1 = _get_ngram_set(s1, n)
    set2 = _get_ngram_set(s2, n)

    if not set1 and not set2:
        return 0.0

    similarity = len(set1.intersection(set2)) / len(set1.union(set2))
    return 1.0 - similarity

# -----------------------------------

def load_and_sort_classes(classes_path: str) -> List[str] | None:
    """Loads unique, sorted class names from the file."""
    if not os.path.exists(classes_path):
        print(f"❌ Error: File not found at: {classes_path}")
        return None

    try:
        with open(classes_path, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, filter empty strings
            lines = [line.strip() for line in f if line.strip()]

        # Return unique and sorted list
        return sorted(list(set(lines)))

    except Exception as e:
        print(f"❌ Unexpected error reading file: {e}")
        return None

def search_animal_class(query: str, class_list: List[str], threshold: float = 0.5) -> List[Tuple[str, float]]:
    """Searches the list using Jaccard Distance and returns matches below the threshold."""
    if not class_list:
        return []

    results = []

    for class_name in class_list:
        distance = jaccard_distance(query, class_name)

        if distance <= threshold:
            results.append((class_name, distance))

    # Sort by distance (best match first)
    results.sort(key=lambda item: item[1])

    return results

# -------------------------- EXECUTION EXAMPLE --------------------------

CLASSES_FILE = "data/animals_names.txt"

sorted_classes = load_and_sort_classes(CLASSES_FILE)