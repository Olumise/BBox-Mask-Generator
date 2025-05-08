import os
import sys
import json
from PIL import Image, ImageDraw
import requests
from io import BytesIO

def create_face_mask(image_path, bbox, output_path=None, padding=0.1):
    """
    Create a black and white mask for face inpainting based on bounding box coordinates.
    
    Args:
        image_path: Path to the image or URL
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        output_path: Path to save the mask (optional)
        padding: Optional padding to add around the face (percentage of bbox size)
    
    Returns:
        PIL Image object of the mask
    """
    # Load the image
    if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    elif isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        # Assume it's already a file-like object
        img = Image.open(image_path)
    
    # Create a black mask of the same size as the original image
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Extract coordinates
    x1, y1, x2, y2 = bbox
    
    # Add padding if specified
    if padding > 0:
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img.width, x2 + pad_x)
        y2 = min(img.height, y2 + pad_y)
    
    # Draw a white rectangle for the face area
    draw.rectangle([x1, y1, x2, y2], fill=255)
    
    # Save the mask if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        mask.save(output_path)
        print(f"Mask saved to: {output_path}")
    
    return mask

def main():
    # Check if we have enough arguments
    if len(sys.argv) < 3:
        print("Usage: python generate_mask.py <image_path> '<bbox_json>'")
        print("Example: python generate_mask.py my_image.jpg '{\"bbox\": [313, 203, 762, 830]}'")
        return
    
    # Get the image path
    image_path = sys.argv[1]
    
    # Parse the bbox JSON
    try:
        bbox_data = json.loads(sys.argv[2])
        if isinstance(bbox_data, dict) and 'bbox' in bbox_data:
            bbox = bbox_data['bbox']
            confidence = bbox_data.get('confidence', None)
            if confidence:
                print(f"Face detected with {confidence * 100:.2f}% confidence")
        elif isinstance(bbox_data, list) and len(bbox_data) == 4:
            bbox = bbox_data
        else:
            print("Error: Invalid bbox format. Expected {\"bbox\": [x1, y1, x2, y2]} or [x1, y1, x2, y2]")
            return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for bbox")
        return
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Generate file names based on input image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join('output', f"{base_name}_mask.png")
    
    # Generate the mask
    create_face_mask(image_path, bbox, mask_path)
    
    print("\nUse these files with your Replicate AI inpainting workflow:")
    print(f"image_file: {image_path}")
    print(f"image_mask: {mask_path}")

if __name__ == "__main__":
    main()
