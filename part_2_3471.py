#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import cv2
import matplotlib.pyplot as plt
def region_growing(img, seeds, thresh):
    # Initialize segmented image
    seg_img = np.zeros_like(img)

    # Initialize seed queue
    seed_queue = []
    for seed in seeds:
        seed_queue.append(seed)

    # Loop until seed queue is empty
    while len(seed_queue) > 0:
        # Get next seed point from queue
        curr_seed = seed_queue.pop(0)

        # Check if current seed point is already segmented
        if seg_img[curr_seed] > 0:
            continue

        # Check if current seed point is within threshold range
        if img[curr_seed] < thresh[0] or img[curr_seed] > thresh[1]:
            continue

        # Set current seed point as segmented
        seg_img[curr_seed] = 255

        # Add neighboring pixels to seed queue
        x, y = curr_seed
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                # Check if neighbor is within image bounds
                if i < 0 or i >= img.shape[0] or j < 0 or j >= img.shape[1]:
                    continue
                # Check if neighbor is already segmented or in seed queue
                if seg_img[i, j] > 0 or (i, j) in seed_queue:
                    continue
                # Check if neighbor is within threshold range
                if img[i, j] < thresh[0] or img[i, j] > thresh[1]:
                    continue
                # Add neighbor to seed queue
                seed_queue.append((i, j))

    return seg_img

# Create the image
img = np.zeros((100, 100))
img[20:40, 20:40] = 120
img[60:80, 60:80] = 200

# Add Gaussian noise
mean = 0
variance = 100
sigma = np.sqrt(variance)
gaussian = np.random.normal(mean, sigma, img.shape)
noisy_img = img + gaussian

# Define seed points
seeds = [(30, 30),(70, 70)]

# Define threshold range for pixel values
thresh = (100, 220)

# Apply region growing algorithm for image segmentation
seg_img = region_growing(noisy_img, seeds, thresh)


# Plot the images 
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(noisy_img, cmap='gray')
ax[1].set_title('Noisy Image')
ax[2].imshow(seg_img, cmap='gray')
ax[2].set_title('Segmented Image')
plt.show()


# In[ ]:




