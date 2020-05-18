# colorimage_Segmentation_by_kmeanclustering
Color image segamentation by k-mean clustering, implemented in matlab.

Acknowledgement: The my implementation was inspired by Prof. Hongdong Li, ANU

# K mean clustering
To implement k-mean clustering, the key is to implement the 4 key steps correctly.
1. I mainly use the built-in randi function to randomly select data points to be the centroids of clusters
2. I mainly use some for loops to compute the closest centroid for all data points.
3. given a cluster (some data point indices), I compute the average value of all data point values in all dimension and these average values stands for the next position of the centroid. All clusters are processed similarly so that we can find out the next position for all centroids.
4. repeat step 2 and 3 until convergence, step 2 and step 3 are therefore in the same big while loop, and the while loop can only be broken if all centroids don’t move anymore.

Moreover, in order to later use the clusters easily, I output the result as a cell array, which is a
cell array containing all clusters of data point indices. Later, I can use these data point indices
to segment the image.

# Impact of K and data point coordinates
By changing the value of K, the performance in terms of color segamentation could be significantly affected. 

Note that when doing k mean clustering, you can choose to include or exclude the coordinates of the data points (although excluding would have much better results).

![comparison](https://i.imgur.com/5a67lIo.png)
![comparison](https://i.imgur.com/jj3xoyH.png)
![comparison](https://i.imgur.com/mvGtrBY.png)
![comparison](https://i.imgur.com/VbvtuJQ.png)

# k mean++
K-mean++ is an improved version of normal k-mean clustering, as it's more stable and faster (but segamentation effect would be similar). And it's actually not difficult to implement. To implement k-mean++, the key is that every time choosing a centroid, it needs to be far from the already chosen centroids. To do that, every time we select a centroid, we need to calculate every data point’s distance to its closest centroid. Then the data points with larger shortest distance to its centroid are more likely to be selected as the next centroid, so that eventually all centroids are far from each other.