# RnnClassifier

Repository for the python package/class RnnClassifier.

## How to use

You will find an example of how to use RnnClassifier for NLP tasks in the jupyter notebook RnnClassifier_example.ipynb. In order to use the classes and methods in your system download the python file rnn_utils.py and put it in the same directory as the rest of your code and call it as it is being illustrated.

## RnnClassifier functions

model = RnnClassifier(n_classes, embedding_dimension, tensorboard_dir, output_architecture=[], cell=tf.contrib.rnn.GRUCell(128), activation=tf.nn.relu, learning_rate=0.01, batch_size=100):

* n_classes: number of distinct classes is the output, e.g. for binary classification problems n_classes=2
* embedding_dimension: dimension of the word embedding, e.g. for one-hot encoding embedding_dimension = vocab_size
* tensorboard_dir: the path that the log file for the tensorboard visualization will be stored. Note that the path should always be an empty directory else tensorboard will try to visualize all the log files
* output_architecture: they layer architecture of the fully connected feed forward neural net between the rnn and the output as a list of integers
* cell: the rnn cell to be used
* activation: the activation for the feed forward neural net
* learning_rate: learning rate of the optimizer
* batch_size: batch size during training

model.train(X, Y, seq_length, epochs=10, verbose=False)

* X: np.array(train_data_size, max_length_size, embedding_dimension)
* Y: np.array(train_data_size, n_classes) one-hot encoded
* seq_length: np.array(train_data_size, 1) indicating the length of each sequence in the data
* epochs: number of epochs to be trained, each epoch the data are shuffled
* verbose: during training will output the the accuracy for every 10 batches

model.predict(X, seq_length)

* X: np.array(test_data_size, max_length_size, embedding_dimension)
* seq_length: np.array(test_data_size, 1) indicating the length of each sequence in the data

model.destruct()

Destroys the tensorflow graph that was created
