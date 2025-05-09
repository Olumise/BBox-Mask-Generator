# Face Mask Generator for Replicate AI

This model generates masks for face inpainting based on bounding box coordinates from face detection. It's designed to work with Replicate AI's two-stage image generation workflow.

## Identity Swap (Using This Tool + Your LoRA Model)

1. Get face detection coordinates from your reference image
2. Use this tool to generate a mask for the face area
3. Use the mask with your LoRA-trained model for inpainting

## How to Use

### API Usage

```python
import replicate

# 1. Generate a mask using this model
mask_output = replicate.run(
    "yourusername/face-mask-generator",
    input={
        "image": "https://example.com/your-reference-image.jpg",
        "bbox_json": '{"bbox": [278, 184, 686, 684], "label": "face", "confidence": 0.789}',
        "padding": 10,  # 10% padding around the face area
        "mask_shape": "rectangle"  # or "circle"
    }
)

# 2. Use the mask with your LoRA model for inpainting
inpainting_output = replicate.run(
    "yourusername/your-lora-model",
    input={
        "prompt": "A portrait of YOUR_IDENTITY with glitter cheeks, in Mark style",
        "image_file": "https://example.com/your-reference-image.jpg",
        "image_mask": mask_output["mask"],
        "lora_url": "your-identity-lora",
        "extra_lora": "luma-portrait",
        "lora_scale": 1.0,
        "extra_lora_scale": 1.1
    }
)
```

## Parameters

- **image**: The reference image to create a mask for
- **bbox_json**: JSON string containing bbox coordinates from face detection
- **padding**: Padding percentage to add around the face area (default: 10)
- **mask_shape**: Shape of the mask to generate ("rectangle" or "circle")

## Output

- **mask**: A black and white PNG image where white (255) represents the area to be inpainted
