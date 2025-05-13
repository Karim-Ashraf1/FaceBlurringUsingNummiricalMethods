import os
import cv2
import numpy as np
import time
from scipy import ndimage
from flask import Flask, request, render_template, redirect, url_for, flash
import base64
from werkzeug.utils import secure_filename

# Function to create a Gaussian kernel


def create_gaussian_kernel(size, sigma):
    if size % 2 == 0:
        size += 1

    # Create a grid of coordinates
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the kernel using the Gaussian formula
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    # Normalize the kernel
    return kernel / np.sum(kernel)

# Function to apply Gaussian blur using numerical convolution


def gaussian_blur_convolution(image, kernel_size, sigma):
    # Create the Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Apply convolution to each channel separately if the image is colored
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = ndimage.convolve(image[:, :, i], kernel)
        return result
    else:
        return ndimage.convolve(image, kernel)

# Function to apply Gaussian blur using Fourier transform (analytical method)


def gaussian_blur_fourier(image, kernel_size, sigma):
    # For simplicity and robustness, we'll use scipy's implementation
    # which handles the Fourier transform details correctly
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = ndimage.gaussian_filter(
                image[:, :, i], sigma=sigma)
        return result
    else:
        return ndimage.gaussian_filter(image, sigma=sigma)


app = Flask(__name__)
app.secret_key = 'face_blurring_app'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
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
        __file__), 'haarcascade_frontalface_alt.xml')
    face_detect = cv2.CascadeClassifier(face_cascade_path)
    if face_detect.empty():
        return None, "Error: Could not load face cascade classifier"

    # Detect faces
    face_data = face_detect.detectMultiScale(image, 1.3, 5)

    if len(face_data) == 0:
        return None, "No faces detected in the image"

    # Create copies for each method
    image_cv2 = image.copy()
    image_conv = image.copy()
    image_fourier = image.copy()

    results = {}

    # Method 1: OpenCV's GaussianBlur
    start_time_cv2 = time.time()
    for (x, y, w, h) in face_data:
        cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image_cv2[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        image_cv2[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    cv2_time = time.time() - start_time_cv2
    results['opencv'] = {'image': image_cv2, 'time': cv2_time}

    # Method 2: Custom convolution
    start_time_conv = time.time()
    for (x, y, w, h) in face_data:
        cv2.rectangle(image_conv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image_conv[y:y+h, x:x+w]
        roi = gaussian_blur_convolution(roi, 23, 30)
        image_conv[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    conv_time = time.time() - start_time_conv
    results['convolution'] = {'image': image_conv, 'time': conv_time}

    # # Method 3: Fourier domain
    # start_time_fourier = time.time()
    # for (x, y, w, h) in face_data:
    #     cv2.rectangle(image_fourier, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     roi = image_fourier[y:y+h, x:x+w]
    #     roi = gaussian_blur_fourier(roi, 23, 30)
    #     image_fourier[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    # fourier_time = time.time() - start_time_fourier
    # results['fourier'] = {'image': image_fourier, 'time': fourier_time}

    # Save the results
    image_results = {}
    for method, data in results.items():
        # Convert RGB to BGR for saving
        img_bgr = cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR)
        result_filename = f"{method}_{os.path.basename(image_path)}"
        result_path = os.path.join(
            app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, img_bgr)

        # Convert to base64 for display
        _, buffer = cv2.imencode('.jpg', img_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        image_results[method] = {
            'img_src': f"data:image/jpeg;base64,{img_base64}",
            'time': f"{data['time']:.4f} seconds"
        }

    return image_results, None


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

            # Convert original image to base64 for display
            with open(filepath, 'rb') as img_file:
                orig_img_base64 = base64.b64encode(
                    img_file.read()).decode('utf-8')

            return render_template('result.html',
                                   original_img=f"data:image/jpeg;base64,{orig_img_base64}",
                                   results=results)

    return render_template('index.html')


@app.route('/simple-upload', methods=['GET'])
def simple_upload():
    return render_template('fallback.html')


if __name__ == '__main__':
    app.run(debug=True)
