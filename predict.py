# predict.py

from cog import BasePredictor, Input, Path
from PIL import Image, ImageDraw
import json
import numpy as np
import os
from typing import List, Dict, Any, Optional

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # No model to load for this simple image processing task
        pass

    def predict(
        self,
        image: Path = Input(description="Input image to create mask for"),
        bbox_json: str = Input(
            description="JSON string containing bbox coordinates from face detection. Format: {\"bbox\": [x1, y1, x2, y2], \"label\": \"face\", \"confidence\": 0.789}"
        ),
        padding: int = Input(
            description="Padding percentage to add around the face area",
            default=10,
            ge=0,
            le=50,
        ),
        mask_shape: str = Input(
            description="Shape of the mask to generate",
            default="rectangle",
            choices=["rectangle", "circle"],
        ),
    ) -> Path:
        """Run face mask generation on the input image"""
        
        # Load the image
        img = Image.open(image)
        
        # Parse the bbox JSON
        try:
            # First try to parse as a complete JSON object
            bbox_data = json.loads(bbox_json)
            
            # Extract the bbox coordinates
            if isinstance(bbox_data, dict) and "bbox" in bbox_data:
                bbox = bbox_data["bbox"]
            elif isinstance(bbox_data, list) and len(bbox_data) == 4:
                bbox = bbox_data
            else:
                raise ValueError("Invalid bbox format. Expected {\"bbox\": [x1, y1, x2, y2]} or [x1, y1, x2, y2]")
                
        except Exception as e:
            raise ValueError(f"Error parsing bbox JSON: {str(e)}")
        
        # Create mask
        mask = self.create_face_mask(img, bbox, padding/100, mask_shape)

        # Ensure the mask is in grayscale format
        mask = mask.convert("L")
        
        # Save the mask to a temporary file
        import tempfile
        out_path = Path(tempfile.mkdtemp()) / "mask.png"
        mask.save(str(out_path))
        
        # Return the path directly
        # Replicate will handle displaying it in the UI
        return out_path
    
    def create_face_mask(self, img, bbox, padding=0.1, mask_shape="rectangle"):
        """Create a black and white mask for face inpainting"""
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
        
        # Draw the mask shape
        draw.rectangle([x1, y1, x2, y2], fill=0)  # First ensure area is black
        
        if mask_shape == "circle":
            # Draw a circle/ellipse for the face area
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            radius_x = (x2 - x1) / 2
            radius_y = (y2 - y1) / 2
            
            draw.ellipse(
                [(center_x - radius_x, center_y - radius_y),
                 (center_x + radius_x, center_y + radius_y)],
                fill=255
            )
        else:  # rectangle
            # Draw a rectangle for the face area
            draw.rectangle([x1, y1, x2, y2], fill=255)
        
        return mask
