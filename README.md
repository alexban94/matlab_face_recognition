# MATLAB Face Recognition
Implementation of the Eigenface method as well as FaceNet compatibility in MATLAB for a facial recognition experiement evaluating performance. The Eigenface method was proposed by [Turk and Pentland](https://direct.mit.edu/jocn/article/3/1/71/3025/Eigenfaces-for-Recognition]) in 1991, and FaceNet by [Schroff et al.](https://arxiv.org/abs/1503.03832) in 2015.


Regarding FaceNet, the pretrained model by Hiroki Taniai was used, available from his <a href=https://github.com/nyoki-mtl/keras-facenet>FaceNet repository</a>. As it was implemented in Keras2, when imported into MATLAB, the Lambda layers are not compatible and are replaced by placeholder layers. ``scalesum_lambda_replacement.m`` is a custom layer which is used to replicate the Lambda layer functionality, which performs a simple operation:  
     <p align=center>  Z = X1 + (X2 ∗ scale)  </p>
During inference, the layer takes two inputs X1 and X2, scales X2 by a predefined scale value and sums it with X1.
In ``facenet_script.m``, the Lambda layers are replaced using this custom layer with the appropriate parameters. At the time it was also necessary to add a new output layer for it to function correctly, but this is ignored when running the model - the activations required for computing predictions are taken from the previous layer.  

The dataset used consisted of 100 training images and classes, with only a single image per class in the training set, and 1344 test images. The objective in using FaceNet was to obtain feature vectors of each training and test image by feeding it through the network. Then, for each test vector, the euclidean distance between all 100 training vectors is calculated. The class of the training vector that is closest (shortest distance) is assigned as the class label prediction of the corresponding test image. 

As a one-shot learning problem, FaceNet managed to obtain a 97.99% accuracy on the test data, compared to 34.45% accuracy with Eigenface. Note that preprocessing was applied to the images in both cases.
