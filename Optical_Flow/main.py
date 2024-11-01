from skimage.io import imread
from scipy import signal, ndimage
import numpy as np
import time
import scipy.io as sio
from matplotlib.pyplot import imshow, show, figure
import skimage.transform as tf
import IPython
import flow_vis
image1 = imread('frame1.png',as_gray=True)
image2 = imread('frame2.png',as_gray=True)
figure()
imshow(image1, cmap='gray')
figure()
imshow(image2, cmap='gray')

flow_gt = sio.loadmat('flow_gt.mat')['groundTruth']
flow_image_gt = flow_vis.flow_to_color(flow_gt)
figure()
imshow(flow_image_gt, cmap='gray')
def lukas_kanade(I1, I2, window_size=5):
    
    w = window_size//2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1 = I1/255.  # normalize pixels
    I2 = I2/255.   # normalize pixels
    
    # Define convolution kernels.
    kernel_x = np.array([[-1.,  1.], [-1., 1.]]) / 4. 
    kernel_y = np.array([[-1., -1.], [ 1., 1.]]) / 4.
    kernel_t = np.array([[1., 1.], [1., 1.]]) / 4.
    
    # Compute partial derivatives.
    Ix = signal.convolve2d((I1 + I2), kernel_x, mode='same') #we add the convolution because with high frequencies involved the derivative might change fast (e.g. not be the general derivative in that direction)
    Iy = signal.convolve2d((I1 + I2), kernel_y, mode='same') 
    It = signal.convolve2d((I2 - I1), kernel_t, mode='same') 
    
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    for i in range(w, I1.shape[0] - w):
        for j in range(w, I1.shape[1] - w):

            # obtain partial derivatives for current patch
            px = Ix[i-w:i+w+1,   j - w:j + w + 1].flatten()
            py = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            pt = It[i - w:i + w + 1, j - w:j + w + 1].flatten()
            
            # Compute optical flow.
            b = -pt
            A = np.stack([px, py]).T
            nu = np.linalg.solve(A.T @ A, A.T @ b)
            
            u[i, j] = nu[0]
            v[i, j] = nu[1]

    return u,v
  t = time.time()

u, v = lukas_kanade(image1, image2, window_size=5)

figure()
imshow(flow_image_gt, cmap='gray')

figure()
flow_image = flow_vis.flow_to_color(np.stack([u,v],axis=-1))
imshow(flow_image,cmap='gray')

print('Elaspsed time: ', time.time()-t)

num_layers = 3
downscale = 2

# Construct image pyramids
pyramids1 = tf.pyramid_gaussian(image1, max_layer=num_layers, downscale=downscale, sigma=1, channel_axis=None)
pyramids2 = tf.pyramid_gaussian(image2, max_layer=num_layers, downscale=downscale, sigma=1, channel_axis=None)

u = np.zeros(image1.shape)
v = np.zeros(image1.shape)

for im1,im2 in zip(reversed(list(pyramids1)),reversed(list(pyramids2))):

    h,w = im1.shape

    u = h/u.shape[0] * tf.resize(u, (h,w), order=1)
    v = h/u.shape[0] * tf.resize(v, (h,w), order=1)

    # Warp image.
    yy, xx = np.mgrid[:h, :w]
    rows, cols = yy + v, xx + u
    im1_warp = ndimage.map_coordinates(im1, [rows, cols])

    # Update optical flow.
    du, dv = lukas_kanade(im1_warp, im2, window_size=5)
    u, v = u + du, v + dv

    flow_image = flow_vis.flow_to_color(np.stack([u,v],axis=-1))
    figure()
    imshow(flow_image,cmap='gray')

#predicting the next iteration assuming constant movement
h,w = image1.shape[:2]
yy, xx = np.mgrid[:h, :w]
flow = np.stack([u,v],axis=-1)
for i in range(1,10):
    
    # Extrapolate the next 10 frames, using the flow between img1 and img2 as one step.
    rows = yy + flow[:,:,1]*i
    cols = xx + flow[:,:,0]*i

    image1_warp = ndimage.map_coordinates(image1, [rows, cols])
    imshow(image1_warp,cmap='gray')
    
    IPython.display.clear_output(True)
    show()

