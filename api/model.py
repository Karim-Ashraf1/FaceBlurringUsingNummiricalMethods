import os
import cv2
import numpy as np
from scipy import ndimage

def create_gaussian_kernel(size, sigma):
    if size % 2 == 0:
        size += 1
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def gaussian_blur_convolution(image, kernel_size, sigma):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = ndimage.convolve(image[:, :, i], kernel)
        return result
    return ndimage.convolve(image, kernel)

def gaussian_blur_fourier(image, kernel_size, sigma):
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = ndimage.gaussian_filter(image[:, :, i], sigma=sigma)
        return result
    return ndimage.gaussian_filter(image, sigma=sigma) 