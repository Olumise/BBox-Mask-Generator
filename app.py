import os
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import json
import base64
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return render_template('index.html', error='No image file uploaded')
            
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No image selected')
        
        # Get coordinates
        try:
            x1 = int(request.form.get('x1', 0))
            y1 = int(request.form.get('y1', 0))
            x2 = int(request.form.get('x2', 0))
            y2 = int(request.form.get('y2', 0))
            bbox = [x1, y1, x2, y2]
        except ValueError:
            return render_template('index.html', error='Invalid coordinates')
        
        # Create output directory
        os.makedirs('static/output', exist_ok=True)
        
        # Save the original image
        original_path = os.path.join('static/output', 'original.jpg')
        file.save(original_path)
        
        # Generate mask
        mask_path = os.path.join('static/output', 'mask.png')
        create_face_mask(original_path, bbox, mask_path)
        
        # Convert mask to base64 for display
        with open(mask_path, 'rb') as f:
            mask_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Get original image as base64 too
        with open(original_path, 'rb') as f:
            original_data = base64.b64encode(f.read()).decode('utf-8')
            
        return render_template('index.html', 
                               mask_image=mask_data, 
                               original_image=original_data,
                               mask_path='/static/output/mask.png',
                               original_path='/static/output/original.jpg')
    
    return render_template('index.html')

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
        mask.save(output_path)
    
    return mask

@app.route('/api/generate-mask', methods=['POST'])
def generate_mask_api():
    # Check if the post request has the file part
    if 'image' not in request.files and 'image_url' not in request.form:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get the bbox from the form
    try:
        bbox_str = request.form.get('bbox', '')
        if bbox_str:
            # Handle the format from Replicate AI
            bbox_data = json.loads(bbox_str)
            if isinstance(bbox_data, dict) and 'bbox' in bbox_data:
                bbox = bbox_data['bbox']
            elif isinstance(bbox_data, list) and len(bbox_data) == 4:
                bbox = bbox_data
            else:
                return jsonify({'error': 'Invalid bbox format'}), 400
        else:
            # Default bbox if none provided
            bbox = [int(request.form.get('x1', 0)), 
                   int(request.form.get('y1', 0)), 
                   int(request.form.get('x2', 0)), 
                   int(request.form.get('y2', 0))]
    except Exception as e:
        return jsonify({'error': f'Invalid bbox format: {str(e)}'}), 400
    
    # Create output directory if it doesn't exist
    os.makedirs('static/output', exist_ok=True)
    
    # Process the image
    if 'image' in request.files:
        # From uploaded file
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the original image
        original_path = os.path.join('static/output', 'original.jpg')
        file.save(original_path)
        
        # Generate mask
        mask_path = os.path.join('static/output', 'mask.png')
        mask = create_face_mask(original_path, bbox, mask_path)
        
    elif 'image_url' in request.form:
        # From URL
        image_url = request.form['image_url']
        
        # Generate mask
        mask_path = os.path.join('static/output', 'mask.png')
        mask = create_face_mask(image_url, bbox, mask_path)
        
        # Save the original image too
        response = requests.get(image_url)
        original_path = os.path.join('static/output', 'original.jpg')
        with open(original_path, 'wb') as f:
            f.write(response.content)
    
    # Return paths to the generated files
    return jsonify({
        'mask_path': f'/static/output/mask.png',
        'original_path': f'/static/output/original.jpg'
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('static/output', filename), as_attachment=True)

@app.route('/download_mask', methods=['POST'])
def download_mask():
    mask_data = request.form.get('mask_data', '')
    if not mask_data:
        return redirect(url_for('index'))
    
    # Decode base64 data
    mask_bytes = base64.b64decode(mask_data)
    
    # Create a temporary file
    temp_path = os.path.join('static/output', 'download_mask.png')
    with open(temp_path, 'wb') as f:
        f.write(mask_bytes)
    
    return send_file(temp_path, as_attachment=True, download_name='face_mask.png')

@app.route('/parse-bbox', methods=['POST'])
def parse_bbox():
    try:
        data = request.json
        bbox_str = data.get('bbox_str', '')
        
        # Try to parse the bbox string
        if not bbox_str:
            return jsonify({'error': 'No bbox data provided'}), 400
            
        # Handle the format from Replicate AI
        bbox_data = json.loads(bbox_str)
        if isinstance(bbox_data, dict) and 'bbox' in bbox_data:
            bbox = bbox_data['bbox']
            return jsonify({
                'x1': bbox[0],
                'y1': bbox[1],
                'x2': bbox[2],
                'y2': bbox[3]
            })
        elif isinstance(bbox_data, list) and len(bbox_data) == 4:
            return jsonify({
                'x1': bbox_data[0],
                'y1': bbox_data[1],
                'x2': bbox_data[2],
                'y2': bbox_data[3]
            })
        else:
            return jsonify({'error': 'Invalid bbox format'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error parsing bbox: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
