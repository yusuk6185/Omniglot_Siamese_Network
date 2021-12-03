# Introduction
Siamese networks are neural networks that contain two or more identical subnetwork
components (Bromley et al., 1994). In this report, the Siamese network will be using two or
more Convolutional Neural Network (CNN) as its subnetwork component. The network will
be trained using the Omniglot dataset to predict whether the given inputs belong to the same
equivalence class.

# Methodology
## Experiment Structure
This experiment has been conducted based on the Siamese network in which several CNNs
are combined. Figure 1. shows an overview of the experiment structure used to build and
train a Siamese network.

![image](https://user-images.githubusercontent.com/35501963/144553433-6846a2c6-551a-4605-bd8d-2fd1d4ad39ee.png)

The Experiment has been conducted in the Google Colab environment. The running
environment for experiment includes Python version 3.6.9, tensorflow version 2.3.0 and
keras version 2.4.3.

## Process for Data
### Pre-processing the Omniglot Dataset
The omniglot dataset is loaded and split into training and test datasets. In the
omniglot_dataset method, the images are converted to grayscale and cast to a float 32
datatype.

![image](https://user-images.githubusercontent.com/35501963/144553533-5589c546-0393-42e8-985a-5740c600d1af.png)

The data is then combined in the preprocess_omniglot_dataset method to form four different
datasets, one used for training and three for testing.

![image](https://user-images.githubusercontent.com/35501963/144553563-4620054d-41ca-44b4-9b11-893ac2c1d28b.png)

Data used for training were only taken from the ‘train’ split.
The three datasets for testing are:
1. Data from the ‘train’ split
2. Data from both the ‘train’ and ‘test’ split
3. Data from the ‘test’ split

### Creating Pairs and Triplets
In testing of the siamese network using contrastive loss function, positive and negative pairs
are required as input. The positive pairs are created using images from the same class and
negative pairs were created by using images from different classes.
In testing of the network using triplet loss function, three inputs are required. An anchor
sample, positive sample and negative sample. Two images of the same class were used for
the anchor and positive sample. An image of a different class was used as the negative
sample

## Siamese Network Design
### CNN Layers
A Convolutional Neural Network (CNN) can take an input and assign importance to various
features of the image and be able to distinguish one from another (Saha, 2018). It uses
filters and layers to process the input image in order to get a good prediction.

![image](https://user-images.githubusercontent.com/35501963/144553649-16bbdcd2-2aea-49db-9d78-ee4f79c963a3.png)

For this experiment, four 2D convolutional layers are used. After each layer, batch
normalisation is applied to maintain the output between 1 and 0. Max Pooling layers were
used after the second and fourth convolutional layer for downsampling the input. Lastly, the
network is connected by two fully connected layers.

Euclidean distance is used to measure the distance between the two outputs from the CNN
model with contrastive loss function. RMSprop is used as an optimizer for the model.

![image](https://user-images.githubusercontent.com/35501963/144553690-fd3df2f4-6ed3-4034-ac9b-af0ce94fdee9.png)

### Loss Function
During the training of the Siamese network, two inputs are encoded and their outputs are
compared. The network will be using two different loss functions to carry out the comparison.
#### Contrastive Loss Function
The contrastive loss function operates a pair of inputs received from the model and
measures the similarity and distance between them. Positive pairs with small distance
between them and negative pairs with distance smaller than the margin m will be learned by
the network. The m value is set at 1 because it performs best in this network.

#### Triplet Loss Function
Triplet loss function uses three inputs instead of pairs: a baseline sample, positive sample
and a negative sample. The function calculates the distance between the positive sample
and the baseline sample and the sum of the distance between the negative and baseline
samples and margin m. The distance is minimized for the positive samples and maximized
for the negative samples. After training, the positive samples will be closer to the baseline
and the negative samples further away. The m value is set at 0.4 for this function.

## Test Design
In order to test the performance of the Siamese network model generated, three
experiments for two different losses, contrastive loss and triplet loss, are conducted. The
three experiments used different pairs as input of the model:
● First experiment used the pairs from the set of glyphs from the ‘train’ split.
● Second experiment used the pairs from the set of glyphs from both ‘train’ and ‘test’
splits.
● Third experiment used the pairs from the set of glyphs from the ‘test’ split.
On the other hand, model training is performed using only ‘train’ split.
Each experiment followed the performance evaluation method to observe the calculated
losses and accuracies of training and validation. Also, a callback function is applied to avoid
overfitting of the model, which is set to stop the experiment if the validation loss does not
change for twenty epochs.

# Results
## Contrastive Loss Function.
### Execution Time & Loss
For three experiments of Siamese Network using contrastive loss function, the training loss,
validation loss and the time taken are computed for fifty epochs and displayed in both table
and plot chart. The figures below display the overall performance evaluation record.

![image](https://user-images.githubusercontent.com/35501963/144553784-97fea052-2bd8-4791-a77c-c434aa04cab7.png)

The first experiment indicated as ‘Contrastive Test 1’ took six seconds for each epoch. The average of validation loss is rounded to 0.072 with the minimum value of 0.049 and max value of 0.1566. On the other hand, the average of training loss is rounded to 0.059 with the minimum value of 0.0336 and maximum value of 0.1601. The trend of chart shows
decreasing loss for both losses throughout the epochs. 

For the second experiment indicated as ‘Contrastive Test 2’, it took seven seconds for each epoch. The average of validation loss is rounded to 0.1048 with the minimum value of 0.0903 and maximum value of 0.1586. Meanwhile, the average of training loss is rounded to 0.0624 with the minimum value of 0.0372 and maximum value of 0.1635. The trend of chart shows decreasing loss for both losses throughout the epochs.

Lastly, for the third experiment indicated as ‘Contrastive Test 3’, it took five seconds for each epoch. The experiment ended after epoch 27 by the callback function. The average of validation loss is rounded to 0.156 with the minimum value of 0.1476 and maximum value of 0.1662. Meanwhile, the average of training loss is rounded to 0.08 with the minimum value of 0.0479 and maximum value of 0.1613. The trend of the chart shows decrease for training loss but constant for validation loss throughout the epochs.

### Accuracy
In addition to the execution time and loss, the training accuracy and validation accuracy for
each epoch using contrastive loss are displayed in the following plot chart.

![image](https://user-images.githubusercontent.com/35501963/144553898-95ad8135-d8b8-44ae-b99b-fce503ac5542.png)

The trend for training accuracy showed high performance, with all the experiments recorded
higher than 95% accuracy before the expiration. After running 50 epochs, the training
accuracies of the final result were 97.12%, 96.72% and 95.41% and the validation
accuracies were 94.29%, 87.65% and 77.03% for the three respective experiments. The
experiment with test split showed comparatively lower validation accuracy.

## Triplet Loss Function
### Execution Time & Loss
For the three experiments of Siamese Network using triplet loss function, the training loss,
validation loss and the time taken are computed for fifty epochs and displayed in both table
and plot chart. The figures below display the overall performance evaluation record.

![image](https://user-images.githubusercontent.com/35501963/144553935-91a42bd5-583f-4b83-b5f2-78a31af40242.png)

The first experiment indicated as ‘Triplet Test 1’ took 4.3 seconds for each epoch. The
average of validation loss is rounded to 0.112 with the minimum value of 0.0834 and max
value of 0.329. On the other hand, the average of training loss is rounded to 0.0594 with the
minimum value of 0.0221 and maximum value of 0.2037. The trend of chart shows
decreasing loss for both losses throughout the epochs.
For the second experiment indicated as ‘Triplet Test 2’, it took 5.3 seconds for each epoch.
The experiment ended after epoch 39 by the callback function. The average of validation
loss is rounded to 0.1513 with the minimum value of 0.1067 and maximum value of 0.396.
Meanwhile, the average of training loss is rounded to 0.0706 with the minimum value of
0.0316 and maximum value of 0.2086. The trend of chart shows decreasing loss for both
losses throughout the epochs.
Lastly, for the third experiment indicated as ‘Triplet Test 3’, it took 4.3 seconds for each
epoch. The experiment ended after epoch 35 by the callback function. The average of
validation loss is rounded to 0.188 with the minimum value of 0.139 and maximum value of
0.392. Meanwhile, the average of training loss is rounded to 0.0731 with the minimum value
of 0.0313 and maximum value of 0.2015.
### Accuracy
In addition to the execution time and loss, the training accuracy and validation accuracy for
each epoch using triplet loss are displayed in the following plot chart.

![image](https://user-images.githubusercontent.com/35501963/144553988-a9192897-a387-4110-a1b3-ab2ab38cef2a.png)

The trend for training accuracy showed high performance, with all the experiments recorded
higher than 99% accuracy before the expiration. After running 50 epochs, the training
accuracies of the final result were 99.77%, 99.58% and 99.64% and the validation
accuracies were 96.34%, 94.55% and 92% for the three respective experiments. The
experiment with test split showed comparatively lower validation accuracy.

## Performance Comparison
The training accuracy and validation accuracy of both contrastive loss and triplet loss are
compared in the figures below.

![image](https://user-images.githubusercontent.com/35501963/144554049-5bd45571-bc4c-4f69-aa78-584976ab8eb7.png)

# Conclusion
It was confirmed that the triplet loss function (TLF) showed better performance than the
contrastive loss function (CLF) in classification of Omniglot image data using the Siamese
network.

TLF showed better accuracy not only in the average, but also in all cases. This is especially
evident in Test 3, using only the test split, TLF showed relatively superior performance
compared to CLF. In addition, the time and accuracy required for training and validation
using TLF is shorter than CLF. Moreover, it was practically confirmed that the predictive
model using triplet is very effective in image classification, especially for classifying various
types of different images such as the Omniglot dataset from the results of this experiment .

In conclusion, TLF is a better loss function to train a Siamese network for the classification of
the Omniglot dataset.

# References
Bromley, J., Guyon, I., LeCum, Y., Sackinger, E., Shah, R. (1994). Signature Verification
using a “Siamese” Time deal Neural Network”

Saha, Sumit. (2018). A Comprehensive Guide to Convolutional Neural Networks — the ELI5way.
Retrieved from https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
