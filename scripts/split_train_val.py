import os
import shutil

def main():
    val_path_yolo = 'datasets/data-yolo/LP/images/val'
    train_path = 'data/train'
    val_new_path = 'data/val'

    # Get validation filenames (without extension)
    val_filenames = [os.path.splitext(file)[0] for file in os.listdir(val_path_yolo) if os.path.isfile(os.path.join(val_path_yolo, file))]

    # Create the new validation folder if it doesn't exist
    os.makedirs(val_new_path, exist_ok=True)

    # Iterate through train folder files
    for filename in os.listdir(train_path):
        basename = os.path.splitext(filename)[0]

        # If the filename is in validation set
        if basename in val_filenames:
            # Source path of the file in train folder
            src_path = os.path.join(train_path, filename)

            # Destination path in new validation folder
            dst_path = os.path.join(val_new_path, filename)

            # Move the file
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to validation folder")

if __name__ == "__main__":
    main()
