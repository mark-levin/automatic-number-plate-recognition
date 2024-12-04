import os
import json
import pandas as pd

def extract_true_labels_from_json(json_path):
    """
    Extract true license plate labels from a single JSON file.
    """
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        true_labels = []
        
        for lp in data.get('lps', []):
            true_labels.append({
                'filename': os.path.basename(json_path).replace('.json', '.jpg'),
                'true_lp_id': lp.get('lp_id', ''),
                'true_lp_text': ''.join([char['char_id'] for char in lp.get('characters', [])])
            })
        
        return true_labels
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return []

def batch_extract_true_labels(json_directory):
    """
    Extract true labels from all JSON files in a directory.
    """
    all_true_labels = []
    
    for filename in os.listdir(json_directory):
        if filename.endswith('.json'):
            json_path = os.path.join(json_directory, filename)
            labels = extract_true_labels_from_json(json_path)
            all_true_labels.extend(labels)
        
    return pd.DataFrame(all_true_labels)
