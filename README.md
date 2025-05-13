# Face Blurring Web Application

This web application demonstrates three different methods for blurring faces in images:

1. OpenCV's built-in GaussianBlur
2. Custom numerical convolution
3. Fourier domain filtering

The application allows users to upload an image, detects faces in the image, and applies all three blurring methods. It then displays the results side by side with timing information.

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the face detection model:
   ```
   curl -o haarcascade_frontalface_alt.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Open your web browser and navigate to http://127.0.0.1:5000/
3. Upload an image containing faces
4. View the results of the three different blurring methods

## How It Works

- Face detection is performed using OpenCV's Haar Cascade Classifier
- The detected face region is extracted and blurred using each method
- The blurred region is then reapplied to the original image
- Timing information is recorded to compare the performance of each method

## Numerical Methods

### OpenCV GaussianBlur

Uses OpenCV's optimized implementation of Gaussian blur

### Custom Numerical Convolution

Implements Gaussian blur using direct convolution with a manually created Gaussian kernel

### Fourier Domain Filtering

Applies Gaussian blur in the frequency domain using the Fourier transform
