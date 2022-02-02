# MATLAB Face Recognition
Implementation of the Eigenface method as well as FaceNet compatibility in MATLAB for a facial recognition experiement evaluating performance. The Eigenface method was proposed by [Turk and Pentland](https://direct.mit.edu/jocn/article/3/1/71/3025/Eigenfaces-for-Recognition]) in 1991, and FaceNet by [Schroff et al](https://arxiv.org/abs/1503.03832) in 2015.


The dataset consisted of 100 training images and classes, with only a single image per class in the training set, and 1344 test images. 

## Eigenface
 Prior to computation of the eigenfaces, the training images are cropped to focus more on the face, resized and standardized. Finally, difference of gaussians (DoG) is peformed to enhance the visibility of features and suppress noise.

The processed training images are vectorized and stored as a matrix, and the average face, *ψ*, is calculated by taking the mean across each row. *ψ* is subtracted from each column to form matrix *A*.



## FaceNet compatibility
