import os
import easyocr
import pandas as pd
from fast_plate_ocr import ONNXPlateRecognizer
from tqdm.notebook import tqdm

def perform_ocr(split='val', processed=False, output_csv=False, engine='easyocr'):
    """
    Perform OCR on cropped images using EasyOCR and optionally save results to a CSV.
    """
    # Input directory for cropped images
    input_dir = f"cropped_images_processed/{split}" if processed else f"cropped_images/{split}"
    
    if engine == 'easyocr':
        # Initialize EasyOCR Reader once
        reader = easyocr.Reader(['en'])  # Modify the language parameter as needed

    # List to store OCR results
    ocr_results = []
    
    # Iterate through cropped images
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            # Full path to the image
            image_path = os.path.join(input_dir, filename)
            
            try:
                if engine == 'easyocr':
                    reader = easyocr.Reader(['en'])
                    result = reader.recognize(image_path, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
                    ocr_text = result[0][1].replace(' ', '').strip()
                
                elif engine == 'fast_plate_ocr':
                    m = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model')
                    result = m.run(image_path)
                    ocr_text = result[0].rstrip('_')
                
                # Store results
                ocr_results.append({
                    'filename': filename,
                    'ocr_text': ocr_text
                })
                
                print(f"OCR for {filename}: {ocr_text}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Convert to DataFrame
    df_ocr = pd.DataFrame(ocr_results)
    
    # Optionally save to CSV
    if output_csv:
        # Ensure output directory exists
        os.makedirs("ocr_results", exist_ok=True)
        
        csv_path = f"ocr_results/ocr_results_{split}.csv"
        df_ocr.to_csv(csv_path, index=False)
        print(f"OCR results saved to {csv_path}")
    
    return df_ocr

def extract_original_filename(cropped_filename):
    """
    Extract the original image filename from the cropped image filename.
    """
    if '_crop_' in cropped_filename:
        original_filename = cropped_filename.split('_crop_')[0] + '.jpg'
    else:
        original_filename = cropped_filename
    return original_filename
