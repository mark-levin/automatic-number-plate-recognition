import argparse
import pandas as pd
from src.data_processing import batch_extract_true_labels
from src.ocr_utils import perform_ocr, extract_original_filename

def main(split, engine):
    # Perform OCR on the specified split with the given OCR engine
    print(f"Performing OCR for split='{split}' using engine='{engine}'...")
    pred_labels = perform_ocr(split=split, processed=True, engine=engine)
    
    # Load true labels
    json_directory = f"data/{split}"
    print(f"Extracting true labels from '{json_directory}'...")
    true_labels = batch_extract_true_labels(json_directory)
    
    # Add a column for the original filenames in predicted labels
    pred_labels['original_filename'] = pred_labels['filename'].apply(extract_original_filename)
    
    # Merge predicted labels with true labels based on the original filename
    merged_df = pd.merge(pred_labels, true_labels, left_on='original_filename', right_on='filename', how='inner')
    
    # Compute OCR accuracy
    merged_df['correct'] = merged_df['ocr_text'] == merged_df['true_lp_text']
    accuracy = merged_df['correct'].sum() / pred_labels.shape[0]
    
    print(f"OCR Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform OCR and calculate accuracy.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split (e.g., 'val', 'test').")
    parser.add_argument("--engine", type=str, required=True, help="OCR engine to use (e.g., 'easyocr').")
    
    args = parser.parse_args()
    main(split=args.split, engine=args.engine)
