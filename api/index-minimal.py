import os
import sys
import time
import base64
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter

# Create Flask app
app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.secret_key = 'face_blurring_app'

# Configure upload folders
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['RESULT_FOLDER'] = '/tmp/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_image(image_path):
    try:
        # Open image using PIL instead of OpenCV
        start_time = time.time()
        img = Image.open(image_path)

        # Apply a simple blur to the entire image since we don't have face detection
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=15))

        # Save the blurred image
        result_filename = f"blurred_{os.path.basename(image_path)}"
        result_path = os.path.join(
            app.config['RESULT_FOLDER'], result_filename)
        blurred_img.save(result_path)

        processing_time = time.time() - start_time

        # Convert original image to base64
        with open(image_path, 'rb') as img_file:
            orig_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Convert blurred image to base64
        with open(result_path, 'rb') as img_file:
            blurred_img_base64 = base64.b64encode(
                img_file.read()).decode('utf-8')

        result = {
            'original': f"data:image/jpeg;base64,{orig_img_base64}",
            'blurred': f"data:image/jpeg;base64,{blurred_img_base64}",
            'time': f"{processing_time:.4f} seconds"
        }

        return result, None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, f"Error processing image: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            # Check if the post request has the file part
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']

            # If user does not select file, browser also submits an empty part without filename
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                results, error = process_image(filepath)
                if error:
                    return jsonify({'error': error}), 500

                return render_template('result-lite.html', results=results)
            else:
                return jsonify({'error': 'File type not allowed'}), 400

        return render_template('index.html')
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/simple-upload', methods=['GET'])
def simple_upload():
    try:
        return render_template('fallback.html')
    except Exception as e:
        print(f"Error in simple_upload route: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Vercel serverless function handler
def handler(environ, start_response):
    try:
        return app(environ, start_response)
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        status = '500 INTERNAL SERVER ERROR'
        response_headers = [('Content-type', 'application/json')]
        start_response(status, response_headers)
        return [jsonify({'error': str(e)}).get_data()]


# For local development
if __name__ == '__main__':
    app.run(debug=True)
