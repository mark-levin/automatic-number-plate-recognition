from ultralytics import YOLO
from PIL import Image
import os

def predict_boxes(model, split, get_cropped_images=True):
    """
    Predict bounding boxes using the YOLO model and optionally crop the images.
    """
    valid_splits = {'train', 'val', 'test'}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}.")

    preds = model.predict(source=f"datasets/data-yolo/LP/images/{split}", device="mps")

    if get_cropped_images:
        # Ensure the output folder for cropped images exists
        output_dir = f"cropped_images/{split}"
        os.makedirs(output_dir, exist_ok=True)

        # Loop through the predictions and images
        for pred in preds:
            # Paths to the downscaled and original images
            downscaled_image_path = pred.path
            image_name = os.path.basename(downscaled_image_path)
            original_image_path = os.path.join("data", split, image_name)

            # Load the downscaled and original images
            downscaled_image = Image.open(downscaled_image_path)
            original_image = Image.open(original_image_path)

            # Get image dimensions
            W_down, H_down = downscaled_image.size
            W_orig, H_orig = original_image.size

            # Calculate scale factors
            scale_x = W_orig / W_down
            scale_y = H_orig / H_down

            # Extract bounding boxes
            boxes = pred.boxes.xyxy
            for i, box in enumerate(boxes):
                # Get bounding box coordinates and adjust them
                x_min, y_min, x_max, y_max = box[:4]
                x_min_orig = int(x_min * scale_x)
                y_min_orig = int(y_min * scale_y)
                x_max_orig = int(x_max * scale_x)
                y_max_orig = int(y_max * scale_y)

                # Crop the original image
                cropped_image = original_image.crop((x_min_orig, y_min_orig, x_max_orig, y_max_orig))

                # Save the cropped image
                cropped_image_name = f"{os.path.splitext(image_name)[0]}_crop_{i}.jpg"
                cropped_image.save(os.path.join(output_dir, cropped_image_name))

                print(f"Cropped image saved: {cropped_image_name}")

    return preds
