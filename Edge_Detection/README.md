# Different methods for detecting edges in images
Method 1: 3 different kernels are convolved with the image to determine the gradients.
Method 2: Detect direction of gradients and when we have strong correlation with nearby pixelgradients then we mark that pixel as an edge even if it might not be a strong enough gradient itself.
