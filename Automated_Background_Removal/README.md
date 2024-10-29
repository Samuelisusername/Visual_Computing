****Removes the bluescreen background from the video of a boxer.**** 

We do this by sampling a few of the blue values and then taking the average of those as a **base colour** for the bluescreen.


Then over the entire video we compute the distance of every pixel in every frame to the base blue colour. If, by our own distance metric, the **distance is above a certain threshhold**,
then we mark as forgound. 
