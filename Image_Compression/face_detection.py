import matplotlib.pyplot as plt
import matplotlib.patches as patches

image = imread('FaceDetection.bmp')[:h,:]/255.
h_,w_ = image.shape

errors = []

# iterate over each part of the image
for x in range(w_-w):
    image_crop = image[:,x:x+w].flatten()
    
    # Compression
    centered_image = image_crop - mean;
    compressed_image = eigenvectors.T @ centered_image;
    
    # Decompression
    centered_image = eigenvectors @ compressed_image
    decompressed_image = centered_image + mean

    error = ((image_crop - decompressed_image)**2).sum()
    errors.append(error)
    
    fig, ax = plt.subplots(2,1)
    rect = patches.Rectangle((x,0),w,h,linewidth=3,edgecolor='r',facecolor='none')
    ax[0].imshow(image,cmap='gray')
    ax[0].add_patch(rect)
    ax[1].plot(np.arange(x+1)+w//2,np.array(errors))
    ax[1].set_xlim([0,w_])
    IPython.display.clear_output(True)
    show()

x_best = np.argmin(errors)
fig, ax = plt.subplots(2,1)
rect = patches.Rectangle((x_best,0),w,h,linewidth=3,edgecolor='g',facecolor='none')

ax[0].imshow(image,cmap='gray')
ax[0].add_patch(rect)
ax[1].plot(np.arange(x+1)+w//2,np.array(errors))
ax[1].set_xlim([0,w_])
IPython.display.clear_output(True)
show()
