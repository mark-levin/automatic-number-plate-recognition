import os
import cv2
import matplotlib.pyplot as plt
import random

def display_images_side_by_side(dir1, dir2, sample_size=5):
    """
    Display images from two directories side by side for comparison.
    """
    dir1_files = set(os.listdir(dir1))
    dir2_files = set(os.listdir(dir2))
    common_files = list(dir1_files.intersection(dir2_files))

    # Take a random sample of common files
    sample_files = random.sample(common_files, min(sample_size, len(common_files)))
    
    for filename in sample_files:
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)
        
        img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(path2, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            print(f"Error reading images: {filename}")
            continue
        
        # Convert images to RGB for Matplotlib (OpenCV uses BGR by default)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Display side-by-side
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Comparison: {filename}", fontsize=16)
        
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img1)
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.title("Processed")
        plt.imshow(img2)
        plt.axis("off")
        
        plt.show()
