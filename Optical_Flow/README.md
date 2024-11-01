# Optical Flow with Lukas Kanade

**1:** we compute the optical flow of every pixel in two images assuming small motion, and assuming in a pixels surrounding pixels move in the same direction


**2:** For objects that can move with large motion we use skimage.transform to constuct the sampling pyramid of both images, such that we have the images at many resolutions. 
We predict the movement on coarse grained resolution and use this information to predict more fine grained movement in the higher resolution images further at the bottom of the pyramids.
 
 
**3:** At last we predict future movement of the pixels/objects by assuming constant movement, as in the previous frame. 

***Possible applications:***
1. Video Stabilization
2. Object Tracking
3. SLAM (simultaneous localization and mapping: method used for autonomous vehicles that lets you build a map and localize your vehicle in that map at the same time)
4. Motion Analysis

