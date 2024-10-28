import matplotlib
from ipywidgets import interactive, widgets
from IPython.display import display
from matplotlib.pyplot import imshow
import numpy as np
import mediapy
bluescreen = mediapy.read_video("bluescreen.avi").astype(int)
numframes, height, width, channels = np.shape(bluescreen)
print("Video information:")
print("Number of frames: ", numframes)
print("Frame width (pixels): ", width)
print("Frame height (pixels): ", height)
print("Video array shape: ", np.shape(bluescreen))

mediapy.show_video(bluescreen.astype(np.uint8))  # the type conversion is important.

# We first load the background mask. This is a single image of shape [H, W, 3], with background = 0 and foreground = 1.
umask = mediapy.read_image("mask.bmp")//255
mediapy.show_image(umask.astype(float))  # Q: Why does this not work as integers?

# The first task is to remove the background of the video in each frame. Try to set the color to black first.
# To do this, you have to loop over the frames, and apply the mask to each frame.
bluescreen_without_background = []
for i in range(numframes):
    bluescreen_without_background.append(bluescreen[i] * umask)
# convert list to array
bluescreen_without_background = np.stack(bluescreen_without_background)
mediapy.show_video(bluescreen_without_background.astype(np.uint8), fps=20)

selected_image_locations = list()
selected_image_locations = [(182, 215), (182, 48), (88, 152)]  # preselected by the TAs

def draw_cross(im, cx, cy, l, w, col):
    dimx, dimy = np.shape(im)[:2]
    im[max(0, cx-l):min(cx+l, dimx-1), max(0, cy-w):min(dimy-1, cy+w)] = col
    im[max(0, cx-w):min(dimx-1, cx+w), max(0, cy-l):min(dimy-1, cy+l)] = col


def display_selections(x, y):
    tmp_im = np.array(bluescreen[0]*umask)
    for px, py in selected_image_locations:
        draw_cross(tmp_im, px, py, 1, 5, (0, 255, 0))
    draw_cross(tmp_im, x, y, 1, 5, (255, 0, 0))
    imshow(tmp_im)

def remember_location(x, y):
    selected_image_locations.append((x, y))

interactive_plot = interactive(display_selections,{'manual': True, 'manual_name': 'Update image'}, x=(0, height-1, 1), y=(0, width-1, 1))
button = widgets.Button(description="Remember this point")
xslider = interactive_plot.children[0]
yslider = interactive_plot.children[1]
button.on_click(lambda b: remember_location(xslider.value, yslider.value))
display(button)
display(interactive_plot)
# A few basic code hints this time, just in case it is the very first time you code in Python :)
# of course this is not a template for the 'best' solution, you are encouraged to change everything you want!

# First we take the x and y pixel coordinates of the first selection made at the previous step. 
# Notice that selected_image_location[0] is a tuple of 2 elements, which we can assign to two scalar variables at once. If the dimensions don't match (like assigning a couple to three variables), Python will throw an error
selx, sely = selected_image_locations[0]
print(f"Selected location: x={selx}, y={sely}")

# get the [r0, g0, b0] values that model our background in this simple single-pixel model
ref_rgb = bluescreen[0, selx, sely]
print(f"Selected rgb: r={ref_rgb[0]}, g={ref_rgb[1]}, b={ref_rgb[2]}")

# set a threshold, you can tune it experimentally after seeing the results
THRESHOLD = 180

# initialize the background mask, it will need to be a 0-1 array of the same sizes of our video frames, so that each 0-1 value corresponds to a pixel being assigned to class background (0) or foreground (1)
bgmask = np.array(umask) # HxWx3 # np.ones_like(bluescreen[0, :, :, :])
# now classify each pixel according to the single-pixel model
for h in range(height):
    for w in range(width):
#         print(h, w, bluescreen[0, h, w], ref_rgb, bluescreen[0, h, w] - ref_rgb, np.linalg.norm(bluescreen[0, h, w] - ref_rgb))
        if np.linalg.norm(bluescreen[0, h, w] - ref_rgb) < THRESHOLD:
            bgmask[h, w, :] = 0.0
# alternatively you can also do it with one line of code using conditional indexing:
bgmask[np.linalg.norm(bluescreen[0] - ref_rgb, axis=-1) < THRESHOLD] = 0

# and visualize your results on the first frame. Do they look reasonable?
imshow(bluescreen[0]*bgmask)
# We can also visualize the residuals
import matplotlib.pyplot as plt
imshow(np.linalg.norm(bluescreen[0] - ref_rgb[None, None], axis=-1) * umask.any(-1))
plt.colorbar()

# Wrap masking in a function
def detect_bg(i):
    bgmask = np.array(umask)
    bgmask[np.linalg.norm(bluescreen[i] - ref_rgb, axis=-1) < THRESHOLD] = 0
    return bgmask

play = widgets.Play(
    interval=1000,
    min=0,
    max=numframes-1,
    step=1,
)

# mask each frame
def print_frame(i):
    bg = detect_bg(i)
    # We want to see the background just a bit, so we keep 25% of its brightness!
    float_mask = bg + 0.25 * (1 - bg)
    imshow(bluescreen[i] / 255. * float_mask)
    
slider = interactive(print_frame, i=(0, numframes-1, 1))
widgets.jslink((play, 'value'), (slider.children[0], 'value'))
widgets.VBox([play, slider])
# Use this time the Mahalanobis distance for the thresholding, with a global mean and covariance of the background
assert(len(selected_image_locations) > 1)
print(selected_image_locations)
exemplar_pixel_set = np.array([bluescreen[0, selx, sely] for selx, sely in selected_image_locations])  # the [`expression` for `element` in `iterator`] list constructor is just an equivalent pythonic short-hand for a for-loop
print(exemplar_pixel_set)
mean = np.mean(exemplar_pixel_set, axis=0)
cov = np.cov(exemplar_pixel_set.T)
inv_cov = np.linalg.pinv(cov)  # you can also use np.linalg.inv here
THRESHOLD = 15

def detect_bg_mah_loop(i):
    """
    Compute the background mask using the mahalanobis distance.
    
    This function is perfectly equivalent to detect_bg_mah_noloop, but uses explitic python looping
    """
    bgmask = np.array(umask)
    
    for h in range(height):
        for w in range(width):
            residual = bluescreen[i, h, w]-mean
            if residual.T @ inv_cov @ residual < THRESHOLD**2:
                bgmask[h, w] = 0

    return bgmask

def detect_bg_mah_noloop(i):
    """
    Compute the background mask using the mahalanobis distance.
    
    This function is perfectly equivalent to detect_bg_mah_loop, but uses implicit numpy broadcasting rules (gains in performance)
    """
    bgmask = np.array(umask)
    
    residuals = bluescreen[i]-mean  # precompute all differences from the mean value
    distance_image = np.sum((residuals[:, :, np.newaxis, :] @ inv_cov)[:, :, 0, :] * residuals, axis=-1)  # build an image where each pixel is the mahalanobis distance from the corresponding source pixel. See docpage of matmul for details on its specific broadcasting rules.
    bgmask[distance_image < THRESHOLD**2] = 0  # conditional assignment of mask values based on the precomputed distance image.
    return bgmask

# and visualize your results on the first frame. Do they look reasonable?
imshow(bluescreen[0]*detect_bg_mah_loop(0))

play = widgets.Play(
    interval=1000,
    min=0,
    max=numframes-1,
    step=1,
)

def print_frame(i):
    bg = detect_bg_mah_loop(i)
    # We want to see the background just a bit, so we keep 25% of its brightness!
    float_mask = bg + 0.25 * (1 - bg)
    imshow(bluescreen[i] / 255. * float_mask)
    
slider = interactive(print_frame, i=(0, numframes-1, 1))
widgets.jslink((play, 'value'), (slider.children[0], 'value'))
widgets.VBox([play, slider])
bgvid = mediapy.read_video("jugglingBG.avi").astype(np.int16)
testvid = mediapy.read_video("jugglingTest.avi").astype(np.int16)

bgnumframes, bgheight, bgwidth, bgchannels = np.shape(bgvid)
print("jugglingBG.avi video information:")
print("Number of frames: ", bgnumframes)
print("Frame width (pixels): ", bgwidth)
print("Frame height (pixels): ", bgheight)
print("Video array shape: ", np.shape(bgvid))
mediapy.show_video(bgvid.astype(np.uint8))

print()
testnumframes, testheight, testwidth, testchannels = np.shape(testvid)
print("jugglingTest.avi video information:")
print("Number of frames: ", testnumframes)
print("Frame width (pixels): ", testwidth)
print("Frame height (pixels): ", testheight)
print("Video array shape: ", np.shape(testvid))
mediapy.show_video(testvid.astype(np.uint8))
means = np.mean(bgvid, axis=0) # extract the mean value for each pixel by averaging over subsequent frames
inv_cov_matrices = np.empty((bgheight, bgwidth, 3, 3), dtype=float)

progress_bar = widgets.IntProgress(min=0.0, max=bgwidth, description='')
display(progress_bar)
for w in range(bgwidth):
    for h in range(bgheight):
        inv_cov_matrices[h, w] = np.linalg.pinv(np.cov(bgvid[:, h, w, :].T))
    progress_bar.value += 1
  THRESHOLD = 50

def detect_bg_perpixelmodel_loop(i):
    """
    Compute the background mask using the mahalanobis distance.
    """
    bgmask = np.ones(shape=(testheight, testwidth, 1), dtype=np.int16)
    
    for h in range(testheight):
        for w in range(testwidth):
            residual = testvid[i, h, w]-means[h, w]
            if (residual.T @ inv_cov_matrices[h, w] @ residual) < THRESHOLD**2:
                bgmask[h, w] = 0

    return bgmask
play = widgets.Play(
    interval=3000,
    min=0,
    max=testnumframes-1,
    step=1,
)

def print_frame(i):
    imshow(testvid[i]*detect_bg_perpixelmodel_loop(i))
    
slider = interactive(print_frame, i=(0, testnumframes-1, 1))
widgets.jslink((play, 'value'), (slider.children[0], 'value'))
widgets.VBox([play, slider])
