#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
# Create the image
img = np.zeros((100, 100))
img[20:40, 20:40] = 120
img[60:80, 60:80] = 200

# Add Gaussian noise
mean = 0
variance = 100
sigma = np.sqrt(variance)
noise = np.random.normal(mean, sigma, img.shape)
noisy_img = img + noise

# Otsu's algorithm implementation
hist, bins = np.histogram(noisy_img, bins=3)
bins = bins[:-1]
pixel_num = np.sum(hist)
mean = np.sum(bins * hist) / pixel_num
max_var = 0
thresh = 0
for i in range(1, 3):
    w0 = np.sum(hist[:i]) / pixel_num
    w1 = np.sum(hist[i:]) / pixel_num
    mu0 = np.sum(bins[:i] * hist[:i]) / (w0 * pixel_num)
    mu1 = np.sum(bins[i:] * hist[i:]) / (w1 * pixel_num)
    var = w0 * w1 * ((mu0 - mu1) ** 2)
    if var > max_var:
        max_var = var
        thresh = i
# Segment the image
segmented_img = np.zeros_like(noisy_img)
segmented_img[noisy_img >= thresh] = 1
        
        
# Plot the images 
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(noisy_img, cmap='gray')
ax[1].set_title('Noisy Image')
ax[2].imshow(segmented_img, cmap='gray')
ax[2].set_title('Segmented Image')
plt.show()


# In[ ]:




