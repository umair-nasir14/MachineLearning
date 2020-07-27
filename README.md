# MachineLearning

This repositary is for all machine learning codes that I am learning from freecodecamp.org.
This starts from Linear Regressing and will go upto Neural Networks.
I am using Anaconda3 and Spyder IDE with Pythoon3.7. I am using Jupyter notebook as well.

Linear Regressing is done with titanic dataset.
https://storage.googleapis.com/tf-datasets/titanic/train.csv
https://storage.googleapis.com/tf-datasets/titanic/eval.csv

Classification uses DNN classifier from tensoflow. The dataset is iris species dataset
https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv
https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv

Hidden Markov Model nor uses a dataset neither do you train a model. It predicts using probability distributions. 
To use tensorflow_probability module you might have to download it through:
$ pip install tensorflow_probability==0.8.0rc0 --user --upgrade

Convolutional Neural Networks:

First model in the file CNN_CIFAR10 is a model that predicts Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship and Truck.
The dataset is 60000 image large with 6000 per catagory. This dataset is built-in Keras dataset known as CIFAR-10.

Natural Language Processing and Recurring Neural Networks:

The file RNN_review_classifier uses a built-in keras dataset, i.e. IMDB dataset, to undertand how sentiment analysis works. Here we are classifying a positive and a negative review.
