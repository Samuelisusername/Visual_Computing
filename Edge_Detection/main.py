from skimage.io import imread
from scipy.ndimage import convolve, shift, label
from matplotlib.pyplot import imshow
from ipywidgets import interactive
from IPython.display import display
import matplotlib as mpl
import numpy as np

mpl.rc('image', cmap='gray')  # tell matplotlib to use gray shades for grayscale images
test_im = np.array(imread("lighthouse.png", as_gray=True), dtype=float)  # This time the image is floating point 0.0 to 1.0!
height, width = test_im.shape
print("Test image shape: ", test_im.shape)
imshow(test_im)
def gaussian_filter(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

slider = interactive(lambda size, sigma: imshow(gaussian_filter(size, sigma)), {'manual': True, 'manual_name': 'Update image'}, size=(1, 49, 2), sigma=(0.1, 20, 0.1))
slider.children[0].value = 9
slider.children[1].value = 2
display(slider)
def thresholded_edge_detection(k_x, sqr_threshold=0.05):
    assert(k_x.ndim == 2)
    k_y = k_x.T
    g_x = convolve(test_im, k_x, mode="constant")
    g_y = convolve(test_im, k_y, mode="constant")
    grad_sqr_mag = g_x**2 + g_y**2
    edges = np.zeros_like(test_im)
    edges[grad_sqr_mag > sqr_threshold] = 1.
    return edges
  basic_k_x = np.array([[-1, 1]])


imshow(thresholded_edge_detection(basic_k_x, sqr_threshold=0.17**2))
prewitt_k_x = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

imshow(thresholded_edge_detection(prewitt_k_x, sqr_threshold=0.6**2))
sobel_k_x = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

imshow(thresholded_edge_detection(sobel_k_x, sqr_threshold=0.8**2))
gauss_k_x = convolve(gaussian_filter(5, 1), np.array([[1, -1]]).T)


imshow(thresholded_edge_detection(gauss_k_x, sqr_threshold=0.045**2))
g_filter = gaussian_filter(9, 1.4)
canny_k_x = convolve(g_filter, np.array([[1, 0, -1]]).T)
canny_k_y = convolve(g_filter, np.array([[1, 0, -1]]))

imshow(canny_k_x)
imshow(canny_k_y)

imshow(thresholded_edge_detection(canny_k_x, sqr_threshold=0.08**2))
g_x = convolve(test_im, canny_k_x)
g_y = convolve(test_im, canny_k_y)

grad_mag = np.sqrt(g_x**2 + g_y**2)

imshow(grad_mag)

grad_dir = np.arctan2(g_y, g_x)  # gradient direction in [-pi, pi]
grad_dir_0_to_8 = 4*(grad_dir/np.pi + 1)  # gradient direction in [0, 8]
quantized_grad_dir = np.array(np.around(grad_dir_0_to_8), dtype=np.uint8)  # map range linearly from [-pi, pi] to [0, 8], then round and make integer index
quantized_grad_dir[quantized_grad_dir == 8] = 0  # both 0 and 3 indicate the same direction (towards negative x) with small y, they belong to the same direction category for us
quantized_grad_dir[quantized_grad_dir > 3] -= 4  # we are interested into the direction, not the side: just merge together the same directions on different sides
# 0: horizontal gradient           (--)
# 1: primary diagonal gradient     (//)
# 2: vertical gradient             (||)
# 3: secondary diagonal gradient   (\\)

imshow(quantized_grad_dir)

comparative_grad_images = np.empty(shape=(4, height, width))
comparative_grad_images[0] = np.maximum(shift(grad_mag, (2, 0)), shift(grad_mag, (-2, 0)))
comparative_grad_images[1] = np.maximum(shift(grad_mag, (2, 2)), shift(grad_mag, (-2, -2)))
comparative_grad_images[2] = np.maximum(shift(grad_mag, (0, 2)), shift(grad_mag, (0, -2)))
comparative_grad_images[3] = np.maximum(shift(grad_mag, (2, -2)), shift(grad_mag, (-2, 2)))
non_maximum_suppressed = np.array(grad_mag)
for i in range(4):
    non_maximum_suppressed[np.logical_and(quantized_grad_dir == i, grad_mag < comparative_grad_images[i])] = 0

imshow(non_maximum_suppressed)
def canny_img(t_high, t_low):
    low_img = np.zeros(shape=grad_mag.shape, dtype=np.int16)
    canny_img = np.zeros(shape=grad_mag.shape, dtype=np.int16)

    low_img[non_maximum_suppressed > t_low] = 1

    labelled_low_img, numlabels = label(low_img, structure=np.ones(shape=(3, 3)))

    canny_img[np.isin(labelled_low_img, list(set(labelled_low_img[non_maximum_suppressed > t_high])))] = 1

    return canny_img

imshow(canny_img(0.1, 0.03))
def perform_all_canny(gauss_size, gauss_sigma, shift_offset, t_high, t_low):
    g_filter = gaussian_filter(gauss_size, gauss_sigma)
    canny_k_x = convolve(g_filter, np.array([[1, 0, -1]]).T)
    canny_k_y = convolve(g_filter, np.array([[1, 0, -1]]))

    g_x = convolve(test_im, canny_k_x)
    g_y = convolve(test_im, canny_k_y)

    grad_mag = np.sqrt(g_x**2 + g_y**2)

    grad_dir = np.arctan2(g_y, g_x)  # gradient direction in [-pi, pi]

    quantized_grad_dir = np.array(np.around(4*(grad_dir/np.pi + 1)), dtype=np.uint8)  # map range linearly from [-pi, pi] to [0, 8], then round and make integer index
    quantized_grad_dir[quantized_grad_dir == 8] = 0  # both 0 and 8 indicate the same direction (towards negative x) with small y, they belong to the same direction category for us
    quantized_grad_dir[quantized_grad_dir > 3] -= 4  # we are interested into the direction, not the side: just merge together the same directions on different sides

    # 0: horizontal gradient           (--)
    # 1: primary diagonal gradient     (//)
    # 2: vertical gradient             (||)
    # 3: secondary diagonal gradient   (\\)

    imshow(quantized_grad_dir)

    comparative_grad_images = np.empty(shape=(4, height, width))
    comparative_grad_images[0] = np.maximum(shift(grad_mag, (shift_offset, 0)), shift(grad_mag, (-shift_offset, 0)))
    comparative_grad_images[1] = np.maximum(shift(grad_mag, (shift_offset, shift_offset)), shift(grad_mag, (-shift_offset, -shift_offset)))
    comparative_grad_images[2] = np.maximum(shift(grad_mag, (0, shift_offset)), shift(grad_mag, (0, -shift_offset)))
    comparative_grad_images[3] = np.maximum(shift(grad_mag, (shift_offset, -shift_offset)), shift(grad_mag, (-shift_offset, shift_offset)))

    non_maximum_suppressed = np.array(grad_mag)
    for i in range(4):
        non_maximum_suppressed[np.logical_and(quantized_grad_dir == i, grad_mag < comparative_grad_images[i])] = 0

    low_img = np.zeros(shape=grad_mag.shape, dtype=np.int16)
    canny_img = np.zeros(shape=grad_mag.shape, dtype=np.int16)

    low_img[non_maximum_suppressed > t_low] = 1

    labelled_low_img, numlabels = label(low_img, structure=np.ones(shape=(3, 3)))

    canny_img[np.isin(labelled_low_img, list(set(labelled_low_img[non_maximum_suppressed > t_high])))] = 1

    return canny_img

slider = interactive(lambda gsz, gsg, so, th, tl: imshow(perform_all_canny(gsz, gsg, so, th, tl)), {'manual': True, 'manual_name': 'Update image'}, gsz=(1, 49, 2), gsg=(0.1, 20, 0.1), so=(0, 5, 0.2), th=(0, 0.4, 0.01), tl=(0, 0.4, 0.01))
slider.children[0].value = 9
slider.children[1].value = 1.4
slider.children[2].value = 2
slider.children[3].value = 0.10
slider.children[4].value = 0.03

display(slider)
