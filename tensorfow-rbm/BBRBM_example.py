#Bernoulli-Bernoulli RBM is good for Bernoulli-distributed binary input data. MNIST, for example.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data

#mnist = tf.keras.datasets.mnist
#(x_train, y_train),(x_test, y_test) = mnist.load_data() 
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
#print(mnist)
mnist_images = mnist.train.images
#print(mnist_images)

bbrbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
errs = bbrbm.fit(mnist_images, n_epoches=5, batch_size=10)
plt.plot(errs)
plt.title('BBRBM Training Results')
plt.ylabel('Train Error')
plt.xlabel('Number of Images Trained On')
plt.show()
