# Compress Images of Faces using Principal Component Analysis (PCA)


By eliminating the least significant set of Eigenvectors in the SVD of the image matrix, we compress the image of faces. 


We can decompress images of faces by taking the eigenface and adding a the compression values times the most significant eigenvectors. 

# Face Detection using the Compression
For every part of the panorama image we compress that part of the panorama and decompress it. If the compression error is small enough then we detect a face. 

This works bease the compression works well exclusively on images that are similar to a face. 
