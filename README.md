# MachineLearningProjects


# Getting Started

This repositary is for all machine learning codes that I am learning from freecodecamp.org.
This starts from Linear Regressing and will go upto Neural Networks.
I am using Anaconda3 and Spyder IDE with Pythoon3.7. I am using Jupyter notebook as well.
Note: I've taken these codes mostly from tensorflow.org and edited it. It is for the understanding the model and applying it to your projects


## Core Machine Learning Algorithms: 

**Linear Regressing** is done with titanic dataset.
https://storage.googleapis.com/tf-datasets/titanic/train.csv
https://storage.googleapis.com/tf-datasets/titanic/eval.csv

**Classification** uses **DNN classifier** from tensoflow. The dataset is iris species dataset
https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv
https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv

**Hidden Markov Model** nor uses a dataset neither do you train a model. It predicts using probability distributions. 
To use tensorflow_probability module you might have to download it through:


'<addr>' $ pip install tensorflow_probability==0.8.0rc0 --user --upgrade

## Convolutional Neural Networks:

The model in the file **CNN_CIFAR10** is a model that predicts Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship and Truck.
The dataset is 60000 image large with 6000 per catagory. This dataset is built-in Keras dataset known as CIFAR-10.

## Natural Language Processing and Recurring Neural Networks:

The file **RNN_review_classifier** uses a built-in keras dataset, i.e. IMDB dataset, to undertand how sentiment analysis works. Here we are classifying a positive and a negative review.

**RNN_play_generator:** is a code that uses RNN model to generate play on the basis of a play given as a data. In this example we have takes the popular play, Romeo and Juliet. NOte the following code allows you to upload any script from your machine to be the data of our model:


'<addr>' from google.colab import files
 
'<addr>'path_to_file = list(files.upload().keys())[0]

