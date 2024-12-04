import os
import cv2
import numpy as np

def deskew_image(image_path, output_path):
    """
    Deskew a single image and save the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Edge detection
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    
    # Detect lines with Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 360, 100)
    
    # Calculate angles of the lines
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)
    
    # Use the median angle as the skew angle
    if len(angles) > 0:
        median_angle = np.median(angles)
    else:
        median_angle = 0  # If no lines are detected, assume no skew
    
    # Rotate the image to correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Save the result
    cv2.imwrite(output_path, rotated)
    print(f"Deskewed image saved: {output_path}")

def process_images(input_dir, output_dir):
    """
    Process images by deskewing, enhancing, and thresholding.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Paths for input and output images
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing {filename}...")
            
            # Deskew the image
            deskew_image(input_path, output_path)
            
            # Read the deskewed image for further processing
            img = cv2.imread(output_path)
            if img is not None:
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize image
                gray = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
                # Apply dilation and erosion
                kernel = np.ones((1, 1), np.uint8)
                gray = cv2.dilate(gray, kernel, iterations=1)
                gray = cv2.erode(gray, kernel, iterations=1)
                # Apply median blur and thresholding
                gray = cv2.medianBlur(gray, 3)
                _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Save the processed image
                cv2.imwrite(output_path, gray)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Could not read deskewed image: {output_path}")
