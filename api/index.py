from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import base64

# Create the Flask app
app = Flask(__name__, template_folder='../templates',
            static_folder='../static')
app.secret_key = 'face_blurring_app'
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create folders if they don't exist (use /tmp for Vercel)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Convert original image to base64 for display
            with open(filepath, 'rb') as img_file:
                orig_img_base64 = base64.b64encode(
                    img_file.read()).decode('utf-8')

            return render_template('fallback.html',
                                   message="Face detection is only available in local development. Please run the application locally for full functionality.",
                                   original_img=f"data:image/jpeg;base64,{orig_img_base64}")

    return render_template('fallback.html',
                           message="Face detection is only available in local development. Please run the application locally for full functionality.")


# For Vercel deployment
app.debug = False
