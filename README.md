# Big Data in Marketing 


The subject of our project was : 
# Image Classification

- With a dataset of food/drinks/brands/logos, create a multilabel image classification model using any pretrained model (e.g. VGG16, InceptionV3, EfficientDet/EfficientNet, etc.)

- Minimum requirements: find/collect and, if needed, annotate enough data for at least 3 classes; use built-in data augmentation functions (e.g. from tensorflow); use transfer learning techniques to train a model; evaluate it on test data; provide and explain how can you apply this model in marketing/market research field to find insights from pictures

- Desirable: Use Albumenatations for image augmentation (example); make some photos and test your model on them; test your model on random pictures of landscapes/inside-outside environments/etc; provide possible marketing ideas of it.


### Explication about the project  

The instructions we got are, with a dataset of images, to create a multilabel classification model using a pretrained model (with at least 3 classes).
We also had to use built-in data augmentation functions from tensorflow, make some photos and test the model on them.  
- Finally, we had to explain how can we apply this model in marketing. 
-------------------------------------------------------------------------------------------------------
Image recognition is often used in marketing projects. In our case, we will focus on logo detection.
For example, it can allow to calculate how much exposure a brand gets from their logo being in images shared across social channels.
Another thing that is often used is to monitor visual conversations about the customer and competitors’ brands of a social media agency using Instagram images -and then join in the chat.
- Image recognition can also be especially useful to recommend products based on the objects seen by a potential customer. In our case, for example, one of the brands could send an offer to a customer who has posted a photo near their restaurant.
--------------------------------------------------------------------------------------------------------
But we will come back to the marketing aspect at the end of the presentation. We will first present the data.
We take a video from several shopping mall at Ankara in Turkey where images were extracted by a group of students at Atilim University. 
We have 6 classes: Subway (153 images), Starbucks (271 images), McDonalds (285 images), KFC (90 images), Burger King (450 images) and other (1049 images). 
The test set will be composed of images that are not relative to the original base but of photos that we will have taken (15 photos per class)., 'KFC',
We will now present our techniques to classify these images. This will take place in 3 steps:
  - With the database we presented just before, we will first use data augmentation function' using Albumentations. 
  - Then we will use transfer techniques, and more specifically use a pre-trained model: InceptionV3. We will also create some layers by our own. 
  - Finally, we will test this model on our 90 photos and compare them. 
Concerning the data pre-processing, first of all we extract the name of the 6 different folders (corresponding to 6 fast foods) to get the labels of our images in the train and the test set.
We can’t write a unique algorithm for each of the condition in which an image is taken, thus, when we acquire an image, we tend to convert it into a form that allows a general algorithm to solve it.
Then, we artificially expand the size of the training set by creating modified versions of images in the dataset. 
This technique is often used for image recognition since a small amount of data is not sufficient to have a robust model. Expand the training dataset will therefore improve the performance and ability of the model to generalize (because this preprocessing is also used to conduct steps that reduce the complexity the applied algorithm).
About this preprocessing technique we used the library Albumentations (a fast and flexible image augmentation library)
This library allows us to do image augmentation that artificially creates training images through different ways of processing or combination of multiple processing, such as random rotation, shifts, shear and flips, etc.
In our case, we apply 5 different filters: we change the color of the image, we flip images, we randomly drop channels in the input image, we crop a random part of the images. We applied a probability to each filter so that an image can have 4 modifications. 
We applied this albumentations technique to our database by using ImageDataAugmentor (a Github library) to apply augmentations pipeline on the training set. 
The usage is analogous to ImageDataGenerator on Keras with the exception that the image transformations will be generated using external augmentations libraries.
So we use this library in order to use folders to import our images with labels.
And, with the help of this library, we also apply a rescaling (which is needed to unified dimensions of images in our dataset), augmentations, a validation split to train. 
Concerning the test set we only rescale it. 
We will now switch to the model that we use for this project. But first, let me do a quick reminder about transfer learning et neural networks. 
The basic idea behind a neural network is to simulate lots of densely interconnected brain cells inside a computer so you can get it to learn things, recognize patterns, and make decisions in a humanlike way. The amazing thing about a neural network is that we don't have to program it to learn explicitly: it learns all by itself, just like a brain.
For the image classification we focused on the convolutional neural network (by definition, it’s neural network containing convolutional layers).
The main operation in a CNN is a convolution, which can be seen as a feature extraction method (in the cases of images: a filtering method). 
Instead of applying a function to an entire image, a convolution scans windows (in our case the scan window of size 3x3) is used and appliy to a kernel (a matrix multiplication typically) to each window. These kernels are seen as image filters. 
The main effect is that it reduces the number of parameters. 

For our CNN we also used transfer learning which consist in reusing a model developed for a task. 
It's currently very popular in deep learning because it can train deep neural networks with comparatively little data as in our case.
In our case, we download the InceptionV3 pre-trained model and its weights from Keras.
More in details, our model is decomposed in 3 steps:
- After taking layers from Inception V3, we freeze them. This avoid destroying any of the information they contain during future training round. 
- Then, we add some new trainable layers on the top of the frozen layers. They will learn to turn the old features into predictions on a new dataset. 
- And finally, we train the layers on our dataset (with an input shape of 224x224x3 (because of colors)). 
So, concerning the new trainable layers, we choose to use:
- The sigmoid function: we use it because it’s between 0 and 1 and, therefore, it is especially used for models where we have to predict the probability as an output.
- Then we add a pooling layer with MaxPooling: the main idea is to “accumulate” features from maps generated by convolving a filter over an image. Formally, its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network. So, max pooling is done to in part to help over-fitting by providing an abstracted form of the representation.
- We also use a dropout method: During training, some number of layer outputs are randomly ignored or “dropped out.” This has the effect of making the layer look-like and be treated-like a layer with a different number of nodes and connectivity to the prior layer. In effect, each update to a layer during training is performed with a different “view” of the configured layer.
- Finally, we use a fully connected layer with relu activation function: ReLU is linear (identity) for all positive values, and zero for all negative values. This means that:
  -	It’s cheap to compute as there is no complicated math. The model can therefore take less time to train or run.
  -	It converges faster. Linearity means that the slope doesn’t saturate, when x gets large.
After creating our model we compile it. It’s an efficiency step since it transforms the simple sequence of layers that we defined into a highly efficient series of matrix.
We used optimization algorithms named ADAM that requires the tuning of learning rate. Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.
Then we train the model. 

------------------------------------------------------------------------------------------------------ 
Concerning the evaluation of the model. If we consider, the accuracy on the training set it is 81% which is sufficient. And you can see in the plots that we have a good accuracy on the validation set (orange curve). 
Nevertheless, the accuracy on our 90 pictures is less conclusive: even if we try to change several parameters of the neural network, it is 30%.
Unfortunately, this can happen because we use totally different pictures from the training set. Indeed, we wanted to get realistic images, thus, blurred pictures, or logos partly hidden by trucks or obstacles.
Now, let’s talk about the table you can see on the right corner. Several metrics can be derived from a confusion matrix and this is what you see in it. For example, recall metrics.
Recall, in other words, the ability of a model is designed to find all the relevant cases within a dataset.
The precise definition of recall is the number of true positives divided by the number of true positives plus the number of false negatives. True positives are data point classified as positive by the model that actually are positive (meaning they are correct), and false negatives are data points the model identifies as negative that actually are positive (incorrect).
R = TP / (TP + FN)
- 47% of images with Starbucks’ logo have been found out.
Precision
An ability of a classifier not to label positive to the negatives
When you say a male is pregnant, which is not possible yet, then this would be detected under this precision score.
(Number of true positive cases) / (Number of all the positive cases)
*all the positive classes = true positive + false positive
The precision and recall metrics are probably the most common metrics derived from such a table.
P = TP / (TP + FP)
- According to the precision, the classifier is better to predict McDonald.


To conclude, we have seen that, to perform logo detection, we have to preprocess the images to get usable training set (we used albumentation to make the training set larger and we rescaled the images to get them more easy to use). Then, as we did, we can use the transfer learning method by taking a pre-trained model (InceptionV3 in our case), freeze its layer and add our own layers to it. After having compiled and trained the model, we can finally get predictions and evaluate the model. In our case, the accuracy on the test set is very poor… However, this kind of framework is very useful in the field of marketing, as we said at the beginning of our presentation.
Hence, as long as we manage to increase the performance of the model, our algorithm could be used to detect the logos of KFC, McDonald’s, etc., on social networks. 
Indeed, a certain company could both detect people who were near their restaurant and propose them offers or see the comments under the picture to respond to them but could also do the same marketing operations with the detection of their competitor's logo.
This algorithm could also be used to see if a marketing operation had an impact by looking at the number of logos appearing on the images before and after. 

