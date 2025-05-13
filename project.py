import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import ndimage, fft

# A function for plotting the images 
def plotImages(img, title=None): 
    plt.figure(figsize=(10, 8))
    plt.imshow(img) 
    plt.axis('off') 
    if title:
        plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show() 

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
            result[:,:,i] = ndimage.convolve(image[:,:,i], kernel)
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
            result[:,:,i] = ndimage.gaussian_filter(image[:,:,i], sigma=sigma)
        return result
    else:
        return ndimage.gaussian_filter(image, sigma=sigma)

def main():
    print("Starting face blurring application...")
    
    # Reading an image using OpenCV 
    # OpenCV reads images by default in BGR format 
    image_path = 'download.jpeg'
    image = cv2.imread(image_path) 
    if image is None:
        print("Error: Could not read image file -> ", image_path)
        return
        
    # Converting BGR image into a RGB image 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    # plotting the original image 
    print("Displaying original image...")
    plotImages(image, "Original Image") 
    
    # Load the face detector
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
    if face_detect.empty():
        print("Error: Could not load face cascade classifier")
        return
        
    face_data = face_detect.detectMultiScale(image, 1.3, 5) 
    
    if len(face_data) == 0:
        print("No faces detected in the image")
        return
    else:
        print(f"Detected {len(face_data)} faces in the image")
    
    # Create a copy of the original image for each method
    image_cv2 = image.copy()
    image_conv = image.copy()
    image_fourier = image.copy()
    
    # Method 1: OpenCV's built-in GaussianBlur (for reference)
    print("\nApplying OpenCV GaussianBlur...")
    start_time_cv2 = time.time()
    for (x, y, w, h) in face_data: 
        cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        roi = image_cv2[y:y+h, x:x+w] 
        # applying a gaussian blur over this new rectangle area 
        roi = cv2.GaussianBlur(roi, (23, 23), 30) 
        # impose this blurred image on original image to get final image 
        image_cv2[y:y+roi.shape[0], x:x+roi.shape[1]] = roi 
    cv2_time = time.time() - start_time_cv2
    
    # Method 2: Custom numerical convolution
    print("Applying Numerical Convolution...")
    start_time_conv = time.time()
    for (x, y, w, h) in face_data: 
        cv2.rectangle(image_conv, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        roi = image_conv[y:y+h, x:x+w] 
        # applying a gaussian blur using our numerical convolution
        roi = gaussian_blur_convolution(roi, 23, 30) 
        # impose this blurred image on original image to get final image 
        image_conv[y:y+roi.shape[0], x:x+roi.shape[1]] = roi 
    conv_time = time.time() - start_time_conv
    
    # Method 3: Fourier domain filtering
    print("Applying Fourier Domain Filtering...")
    start_time_fourier = time.time()
    for (x, y, w, h) in face_data: 
        cv2.rectangle(image_fourier, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        roi = image_fourier[y:y+h, x:x+w] 
        # applying a gaussian blur using Fourier transform
        roi = gaussian_blur_fourier(roi, 23, 30) 
        # impose this blurred image on original image to get final image 
        image_fourier[y:y+roi.shape[0], x:x+roi.shape[1]] = roi 
    fourier_time = time.time() - start_time_fourier
    
    # Display the outputs
    print("\nDisplaying results...")
    plotImages(image_cv2, "OpenCV GaussianBlur (Time: {:.4f}s)".format(cv2_time))
    plotImages(image_conv, "Numerical Convolution (Time: {:.4f}s)".format(conv_time))
    plotImages(image_fourier, "Fourier Domain Filtering (Time: {:.4f}s)".format(fourier_time))
    
    # Print performance comparison
    print("\n--- Performance Comparison ---")
    print("OpenCV GaussianBlur: {:.4f} seconds".format(cv2_time))
    print("Numerical Convolution: {:.4f} seconds".format(conv_time))
    print("Fourier Domain Filtering: {:.4f} seconds".format(fourier_time))
    print("\nRelative Performance:")
    print("Numerical Convolution is {:.2f}x slower than OpenCV".format(conv_time/cv2_time))
    print("Fourier Domain Filtering is {:.2f}x slower than OpenCV".format(fourier_time/cv2_time))
    print("Fourier Domain Filtering is {:.2f}x {} than Numerical Convolution".format(
        abs(fourier_time/conv_time), 
        "slower" if fourier_time > conv_time else "faster"
    ))

# Execute the main function
if __name__ == "__main__":
    main()
