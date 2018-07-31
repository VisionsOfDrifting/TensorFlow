import tensorflow as tf
mnist = tf.keras.datasets.mnist
# ^^^ this command seems to be better in 1.9.0 than:
# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
# ^^^ this command must be for a previous API version

(x_train, y_train),(x_test, y_test) = mnist.load_data() 
# ^^^ returns this exactly. If you look up the function this line of code is the only thing in the documentation

# Create a callback to pass to our data. This is needed to invoke tensorboard
# Just know that multiple runs will become overlayed on the graphs so you have to delete the
# event logs in the log_dir everytime. Do a rm /tmp/ml_test_Graph/* or something for now.
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='/tmp/ml_test_Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

x_train, x_test = x_train / 255.0, x_test / 255.0
# ^^^ normalize the pixel values. x_train[i] is a 28x28 square matrix with values 0-255 for pixel intensities
# so we divide by 255 as a method of feature scalling remember the formula x' = (x-min(x))/(max(x)-min(x))
# here this evaluates to x' = (x-0)/(255-0) so we call all the training examples by 255.
# y_train is the output vector of classes 0-9 because this is not matlab/octave indicies start at 0 instead of 1

# Sequential is a Linear stack of layers.
# layers.Dense(outputSize, activation function)
# relu 'rectified linear unit' -> f(x) = x^+ = max(0,x) This is softplus without negative, range[0,1] instead of [-1,1]
# the softplus function is the integral of the sigmoid function: f(x) = log(1+exp(x))
# softmax is sigmoid function for multi-class classification. There are slight differences, but it is conecptually the same
# softmax -> P(y = j| x) = exp(x'w_j)/sum{k=1 to K}(exp(x'w_k)
# layers.Dropout is a method of scalling input by arg 'rate'. See the details in the documentation
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Sequential.comile is a function, the arguments are scattered in the documentation
model.compile(optimizer='Adam', # you can also specify it like so optimizer=tf.train.AdamOptimizer()
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Check out https://keras.io/optimizers/
# optimizers: "Adadelta, Adagrad, Adam, Adamax, Nadam, RMSProp, SGD"
# Check out https://keras.io/losses/
# loss functions: "mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error,
#   squared_hinge, hinge, categorical_hinge, logcosh, categorical_crossentropy, sparse_categorical_crossentropy, 
#   binary_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity"

#history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

# The default batch_size is 32
# You can also specify batch_size, validation_split, validation_data and more
model.fit(x_train, y_train, epochs=5, callbacks=[tbCallBack])
# You can also specify batch_size and sample_weight.
print(model.evaluate(x_test, y_test))

print("Run the command line:\n" \
          "tensorboard --logdir=/tmp/ml_test_Graph " \
"\nThen open http://0.0.0.0:6006/ into your web browser")