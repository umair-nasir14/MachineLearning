import tensorflow_probability as tfp  
import tensorflow as tf

#defining transitions and observations

initial_distribution = tfp.distributions.Categorical(probs=[0.8, 0.2]) #fisrt day is 80% cold
transition_distribution = tfp.distributions.Categorical(probs=[[0.7, 0.3],
                                                 [0.2, 0.8]])  #Cold day has 30%  chance of being followed by a hot day and a hot day has 20% chance of being followed by a cold day
observation_distribution = tfp.distributions.Normal(loc=[0., 15.], scale=[5., 10.])  #the loc argument represents the mean and the scale is the standard devitation. FIrst is cold and then hot day.


#Creating Hidden Markov Model
model = tfp.distributions.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)


#Finding the probability
mean = model.mean()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())
