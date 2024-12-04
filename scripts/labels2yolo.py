import os
import json
import cv2
import argparse
from tqdm import tqdm
import scripts.utils as utils
import shutil
import random

def create_yolo_bbox_string(class_id, bbox, img_width, img_height):
    x_center = (bbox[0][0] + bbox[1][0]) / (2 * img_width)
    y_center = (bbox[0][1] + bbox[1][1]) / (2 * img_height)
    width = (bbox[1][0] - bbox[0][0]) / img_width
    height = (bbox[1][1] - bbox[0][1]) / img_height
    return f'{class_id} {x_center} {y_center} {width} {height}'

def write_split_files(directory, train_files, val_files, test_files):
    """Write train.txt, val.txt, and test.txt files."""
    # Create paths relative to the images directory
    train_paths = [os.path.join('images/train', f + '.jpg') for f in train_files]
    val_paths = [os.path.join('images/val', f + '.jpg') for f in val_files]
    test_paths = [os.path.join('images/test', f + '.jpg') for f in test_files]
    
    # Write train.txt
    with open(os.path.join(directory, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_paths))
    
    # Write val.txt
    with open(os.path.join(directory, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_paths))
    
    # Write test.txt
    with open(os.path.join(directory, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_paths))

def transform_dataset(input_directory, lp_size, ocr_size):
    train_txt_path = os.path.join(input_directory, 'train.txt')
    test_txt_path = os.path.join(input_directory, 'test.txt')

    # Normalize input directory
    input_directory = os.path.normpath(input_directory)

    last_dir = os.path.basename(os.path.normpath(input_directory)) + '-yolo'
    lp_directory = os.path.join(os.path.dirname(input_directory),
        last_dir, 'LP')                                
    ocr_directory = os.path.join(os.path.dirname(input_directory),
        last_dir, 'OCR')
    
    # Create directories if not exist
    for dataset_dir in [lp_directory, ocr_directory]:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

    ocr_classes = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # Process test set first
    print('Processing test split')
    with open(test_txt_path, 'r') as f:
        test_filenames = [os.path.splitext(os.path.basename(line.strip()))[0] 
                         for line in f.readlines()]
    
    process_files(test_filenames, 'test', 'test', input_directory, lp_directory, 
                 ocr_directory, ocr_classes, lp_size, ocr_size)

    # Process train set and create train/val split
    print('Processing train split')
    with open(train_txt_path, 'r') as f:
        train_filenames = [os.path.splitext(os.path.basename(line.strip()))[0] 
                          for line in f.readlines()]
    
    # Randomly shuffle and split train data
    random.shuffle(train_filenames)
    split_idx = int(len(train_filenames) * 0.8)
    train_subset = train_filenames[:split_idx]
    val_subset = train_filenames[split_idx:]

    # Process train subset
    process_files(train_subset, 'train', 'train', input_directory, lp_directory, 
                 ocr_directory, ocr_classes, lp_size, ocr_size)
    
    # Process validation subset
    process_files(val_subset, 'train', 'val', input_directory, lp_directory, 
                 ocr_directory, ocr_classes, lp_size, ocr_size)

    # Create split files for LP detection
    print('Creating split files for LP detection')
    write_split_files(lp_directory, train_subset, val_subset, test_filenames)

    # Create split files for OCR detection
    print('Creating split files for OCR detection')
    # For OCR, we need to account for multiple license plates per image
    ocr_train_files = []
    ocr_val_files = []
    ocr_test_files = []

    # Helper function to get OCR filenames for a given image
    def get_ocr_filenames(filename, source_split):
        json_path = os.path.join(input_directory, source_split, filename + '.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        return [f'{filename}_{lp["lp_id"]}' for lp in data['lps']]

    # Get OCR filenames for each split
    for filename in train_subset:
        ocr_train_files.extend(get_ocr_filenames(filename, 'train'))
    
    for filename in val_subset:
        ocr_val_files.extend(get_ocr_filenames(filename, 'train'))
    
    for filename in test_filenames:
        ocr_test_files.extend(get_ocr_filenames(filename, 'test'))

    write_split_files(ocr_directory, ocr_train_files, ocr_val_files, ocr_test_files)

def process_files(filenames, source_split, target_split, input_directory, lp_directory, 
                 ocr_directory, ocr_classes, lp_size, ocr_size):
    for filename in tqdm(filenames):
        # Load image
        img_path = os.path.join(input_directory, source_split, filename + '.jpg')
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape

        # Load JSON label
        json_path = os.path.join(input_directory, source_split, filename + '.json')
        with open(json_path, 'r') as f:
            data = json.load(f)

        # License Plate Detection Dataset
        for lp_data in data['lps']:
            lp_img = img.copy()
            lp_id = lp_data['lp_id']
            poly_coord = lp_data['poly_coord']

            # Convert polygonal annotation to rectangular bbox
            lp_bbox = utils.poly2bbox(poly_coord)

            # Write license plate image
            lp_output_path = os.path.join(lp_directory, 'images', 
                target_split, f'{filename}.jpg')
            # Check if file exists
            if not os.path.isfile(os.path.dirname(lp_output_path)):
                rescale_factor_lp = lp_size / max(img_height, img_width)
                # Resize license plate to desired size
                lp_img_resized = cv2.resize(lp_img, (int(img_width * rescale_factor_lp), 
                                                    int(img_height * rescale_factor_lp)))
                cv2.imwrite(lp_output_path, lp_img_resized)

            # Write YOLO bbox annotation for license plate
            lp_yolo_path = os.path.join(lp_directory, 'labels',
                target_split, f'{filename}.txt')

            append_write_lp = 'a' if os.path.exists(lp_yolo_path) else 'w'
            with open(lp_yolo_path, append_write_lp) as lp_f:
                lp_f.write(create_yolo_bbox_string(0, lp_bbox, img_width, img_height) + '\n')

            # Write OCR image
            ocr_output_path = os.path.join(ocr_directory, 'images', 
                target_split, f'{filename}_{lp_id}.jpg')
            # Crop lp_img to lp_bbox
            ocr_img = lp_img[lp_bbox[0][1]:lp_bbox[1][1], lp_bbox[0][0]:lp_bbox[1][0]]
            ocr_img_offset_x = lp_bbox[0][0]
            ocr_img_offset_y = lp_bbox[0][1]
            ocr_height, ocr_width, _ = ocr_img.shape
            # Resize OCR image to desired size
            rescale_factor_ocr = ocr_size / max(ocr_height, ocr_width)
            ocr_img_resized = cv2.resize(ocr_img, (int(ocr_width * rescale_factor_ocr),
                                                   int(ocr_height * rescale_factor_ocr)))
            cv2.imwrite(ocr_output_path, ocr_img_resized)

            # OCR Detection Dataset
            for char_data in lp_data['characters']:
                char_id = char_data['char_id']
                bbox = char_data['bbox_coord']
                bbox = [[bbox[0][0] - ocr_img_offset_x, bbox[0][1] - ocr_img_offset_y],
                        [bbox[1][0] - ocr_img_offset_x, bbox[1][1] - ocr_img_offset_y]]
                class_id = ocr_classes.index(char_id)

                # Write YOLO bbox annotation for character
                ocr_yolo_path = os.path.join(ocr_directory, 'labels', target_split,
                                             f'{filename}_{lp_id}.txt')
                append_write_ocr = 'a' if os.path.exists(ocr_yolo_path) else 'w'
                with open(ocr_yolo_path, append_write_ocr) as ocr_f:
                    ocr_f.write(create_yolo_bbox_string(class_id, bbox, ocr_width, ocr_height) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', type=str, help='Path to input dataset')
    parser.add_argument('lp_size', type=int, help='YOLO input size for LP detection')
    parser.add_argument('ocr_size', type=int, help='YOLO input size for OCR detection')
    args = parser.parse_args()
    
    transform_dataset(args.input_directory, args.lp_size, args.ocr_size)