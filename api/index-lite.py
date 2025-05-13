import os
import sys
import cv2
import numpy as np
import time
import base64
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.secret_key = 'face_blurring_app'
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['RESULT_FOLDER'] = '/tmp/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not read image file"

    # Convert to RGB for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the face detector
    face_cascade_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'haarcascade_frontalface_alt.xml')
    face_detect = cv2.CascadeClassifier(face_cascade_path)
    if face_detect.empty():
        return None, "Error: Could not load face cascade classifier"

    # Detect faces
    face_data = face_detect.detectMultiScale(image, 1.3, 5)

    if len(face_data) == 0:
        return None, "No faces detected in the image"

    # Create a copy for processing
    image_cv2 = image.copy()

    results = {}

    # Apply OpenCV's GaussianBlur
    start_time_cv2 = time.time()
    for (x, y, w, h) in face_data:
        cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image_cv2[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        image_cv2[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    cv2_time = time.time() - start_time_cv2

    # Convert RGB to BGR for saving
    img_bgr = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
    result_filename = f"blurred_{os.path.basename(image_path)}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, img_bgr)

    # Convert to base64 for display
    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Prepare original image
    with open(image_path, 'rb') as img_file:
        orig_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    result = {
        'original': f"data:image/jpeg;base64,{orig_img_base64}",
        'blurred': f"data:image/jpeg;base64,{img_base64}",
        'time': f"{cv2_time:.4f} seconds"
    }

    return result, None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('simple_upload'))

        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results, error = process_image(filepath)
            if error:
                flash(error)
                return redirect(request.url)

            return render_template('result-lite.html', results=results)

    return render_template('index.html')


@app.route('/simple-upload', methods=['GET'])
def simple_upload():
    return render_template('fallback.html')

# Vercel serverless function handler


def handler(environ, start_response):
    return app(environ, start_response)


# For local development
if __name__ == '__main__':
    app.run(debug=True)
