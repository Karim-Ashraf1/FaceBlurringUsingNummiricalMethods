import os 
import cv2 
import numpy as np  
from scipy import ndimage  # for image processing
from flask import Flask, request, render_template, redirect, url_for, flash
import base64  # for converting images to text
from werkzeug.utils import secure_filename  # for making filenames safe


def create_gaussian_kernel(size, sigma):
    # Make sure size is odd number
    if size % 2 == 0:
        size += 1

    # Create a grid of points
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)

    # Make the blur filter using math formula
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    # Make sure all numbers add up to 1
    return kernel / np.sum(kernel)



def gaussian_blur_convolution(image, kernel_size, sigma):
    # Make our blur filter
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # If image is colored (has 3 channels - red, green, blue)
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        # Blur each color separately
        for i in range(3):
            result[:, :, i] = ndimage.convolve(image[:, :, i], kernel)
        return result
    else:
        # If image is black and white, just blur once
        return ndimage.convolve(image, kernel)




def gaussian_blur_fourier(image, kernel_size, sigma):
    # If image is colored
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        # Blur each color separately
        for i in range(3):
            result[:, :, i] = ndimage.gaussian_filter(
                image[:, :, i], sigma=sigma)
        return result
    else:
        # If image is black and white
        return ndimage.gaussian_filter(image, sigma=sigma)


# Create our web app
app = Flask(__name__)
app.secret_key = 'face_blurring_app'  
app.config['UPLOAD_FOLDER'] = 'static/uploads'  
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not read image file"

    # Convert from BGR to RGB (OpenCV uses BGR, but we want RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the face detector
    face_cascade_path = os.path.join(os.path.dirname(
        __file__), 'haarcascade_frontalface_alt.xml')
    face_detect = cv2.CascadeClassifier(face_cascade_path)
    if face_detect.empty():
        return None, "Error: Could not load face detector"

    # Find faces in the image
    face_data = face_detect.detectMultiScale(image, 1.3, 5)

    if len(face_data) == 0:
        return None, "No faces found in the image"

    # Make copies of the image for each blur method
    image_cv2 = image.copy()
    image_conv = image.copy()
    image_fourier = image.copy()

    results = {}

    # Method 1: Use OpenCV's built-in blur
    for (x, y, w, h) in face_data:
        # Draw a green box around the face
        cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Get the face area
        roi = image_cv2[y:y+h, x:x+w]
        # Blur the face
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        # Put the blurred face back
        image_cv2[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    results['opencv'] = {'image': image_cv2}

    # Method 2: Use our custom blur
    for (x, y, w, h) in face_data:
        cv2.rectangle(image_conv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image_conv[y:y+h, x:x+w]
        roi = gaussian_blur_convolution(roi, 23, 30)
        image_conv[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    results['convolution'] = {'image': image_conv}

    # Method 3: Use Fourier transform blur
    for (x, y, w, h) in face_data:
        cv2.rectangle(image_fourier, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image_fourier[y:y+h, x:x+w]
        roi = gaussian_blur_fourier(roi, 23, 30)
        image_fourier[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    results['fourier'] = {'image': image_fourier}

    # Save all the results
    image_results = {}
    for method, data in results.items():
        # Convert back to BGR for saving
        img_bgr = cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR)
        # Save the image
        result_filename = f"{method}_{os.path.basename(image_path)}"
        result_path = os.path.join(
            app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, img_bgr)

        # Convert to base64 for showing in web page
        _, buffer = cv2.imencode('.jpg', img_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        image_results[method] = {
            'img_src': f"data:image/jpeg;base64,{img_base64}"
        }

    return image_results, None



@app.route('/', methods=['GET', 'POST'])
def index():
    # If someone uploaded an image
    if request.method == 'POST':
        # Check if they actually uploaded something
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)

        file = request.files['file']

        # Check if they selected a file
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # If we have a valid image file
        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image
            results, error = process_image(filepath)
            if error:
                flash(error)
                return redirect(request.url)

            # Get the original image for display
            with open(filepath, 'rb') as img_file:
                orig_img_base64 = base64.b64encode(
                    img_file.read()).decode('utf-8')

            # Show the results
            return render_template('result.html',
                                    original_img=f"data:image/jpeg;base64,{orig_img_base64}",
                                    results=results 
                                    )

    # If it's a GET request, show the upload form
    return render_template('index.html')



@app.route('/simple-upload', methods=['GET'])
def simple_upload():
    return render_template('fallback.html')


# Run the app
if __name__ == '__main__':
    app.run(debug=True) 
