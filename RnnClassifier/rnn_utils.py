import gensim
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def pre_processX(X):
    ##zero padding and X to ndarray
    batch_size = len(X)
    lenseq = [len(i) for i in X]
    maxlen = max(lenseq)
    emp_dim = len(X[0][0])
    padded = []
    for seq in X:
        padded_l = seq
        for emp in range(maxlen-len(seq)):
            padded_l.append(emp_dim*[0])
        padded.append(padded_l)
    flattened = np.array([item for sublist in padded for sublist2 in sublist for item in sublist2])
    X_input = flattened.reshape(batch_size,maxlen,emp_dim)
    return X_input, lenseq

def pre_processY(Y):
    Y_input = np.zeros([Y.shape[0], len(Y.unique())])
    Y_input[np.arange(Y.shape[0]),Y] = 1
    return Y_input

def variable_summary(var, name):
    tf.summary.scalar(name, tf.reduce_mean(var))
    tf.summary.histogram(name, var)
    
class RnnClassifier(object):
    def __init__(self, n_classes, embedding_dimension, tensorboard_dir, output_architecture=[], cell=tf.contrib.rnn.GRUCell(128),
                 activation=tf.nn.relu, learning_rate=0.01, batch_size=100):
        tf.reset_default_graph()
        self.cell = cell
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.embedding_dimension = embedding_dimension
        self.output_architecture = output_architecture
        self.activation = activation
        self.tensorboard_dir = tensorboard_dir
        self.seq_length = tf.placeholder(tf.float32, [None])
        self.x = tf.placeholder(tf.float32, [None, None, self.embedding_dimension])
        self.y = tf.placeholder(tf.float32, [None, n_classes])
        self._weights_init()
        self._forward()
        self._cost()
        self._optimize()
        self._accuracy()
        
        self.merged = tf.summary.merge_all()
        self.board_writer = tf.summary.FileWriter(self.tensorboard_dir)
        
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.board_writer.add_graph(self.sess.graph)
        
    def variable_summary(self, var, name):
        tf.summary.scalar(name, tf.reduce_mean(var))
        tf.summary.histogram(name, var)
    
    def _forward(self):
        ####recursion
        with tf.name_scope('RNN'):
            with tf.variable_scope('RNN', initializer=tf.contrib.layers.xavier_initializer()):
                rnn_output, state = tf.nn.dynamic_rnn(cell = self.cell, 
                                              inputs = self.x,
                                            sequence_length=self.seq_length,
                                          dtype=tf.float32)
                self.variable_summary(state, "state")
                self.variable_summary(rnn_output[:,-1,:], "last_layer_output")
                
        ###fully connected output
        with tf.name_scope('Fully_connected_output'):
            layer_input = state
            for i in range(len(self.output_architecture)):
                layer_input = self.activation(tf.add(tf.matmul(layer_input, self.variables["layer_{}_weights".format(i)]), 
                           self.variables["layer_{}_bias".format(i)]))
            self.acceptor = tf.add(tf.matmul(layer_input, self.w_out), self.b_out)
            self.prediction = tf.nn.softmax(self.acceptor)
            self.variable_summary(self.acceptor, "Fully_connected_output")
        
        
    def _weights_init(self):
        self.variables = {}
        Mi = self.cell.output_size
        for layer, i in zip(self.output_architecture, range(len(self.output_architecture))):
            Mo = layer
            self.variables["layer_{}_weights".format(i)] = tf.Variable(xavier_init(Mi, Mo))
            self.variables["layer_{}_bias".format(i)] = tf.Variable(tf.zeros(Mo))
            self.variable_summary(self.variables["layer_{}_weights".format(i)], "layer_{}_weights".format(i))
            self.variable_summary(self.variables["layer_{}_bias".format(i)], "layer_{}_bias".format(i))
            Mi = Mo
        self.w_out = tf.Variable(xavier_init(Mi, self.n_classes))
        self.b_out = tf.Variable(tf.zeros(self.n_classes), tf.float32)
        self.variable_summary(self.w_out, "output_weight")
        self.variable_summary(self.b_out, "output_bias")
        
        
    def _cost(self):
        with tf.name_scope('cross_entropy'):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.acceptor, labels=self.y))
            self.variable_summary(self.cross_entropy, 'cross_entropy')
        return self.cross_entropy
    
    def _optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        return self.optimizer
    
    def _accuracy(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.variable_summary(self.accuracy, 'Accuracy')
            
    def predict(self, X, sequence):
        predictions = self.sess.run(self.prediction, feed_dict={self.x:X, self.seq_length:sequence})
        return predictions
    
    def destruct(self):
        tf.reset_default_graph()
    
    def train(self, X, Y, seq_length, epochs=10, verbose=False):
        N = len(X)
        n_batches = int(N/self.batch_size)
        for i in range(epochs):
            X, Y, seq_length = shuffle(X, Y, seq_length, random_state=0)
            avg_cost=0
            for j in range(0, N, self.batch_size):
                train_X = X[j:j+self.batch_size]
                train_Y = Y[j:j+self.batch_size]
                batch_seq_length = seq_length[j:j+self.batch_size]
                c, _ = self.sess.run([self.cross_entropy, self.optimizer], 
                                     feed_dict={self.x:train_X, self.y:train_Y, 
                                                self.seq_length:batch_seq_length})
                avg_cost +=c
                if j%5==0:
                    run_summ = self.sess.run(self.merged,
                                             feed_dict={self.x:train_X, self.y:train_Y, 
                                                        self.seq_length:batch_seq_length})
                    self.board_writer.add_summary(run_summ, j)
                if (j%1000==0) and verbose:
                    train_acc = self.sess.run([self.accuracy],
                                              feed_dict={self.x:train_X, self.y:train_Y, 
                                                         self.seq_length:batch_seq_length}) 
                    print("Step {} of epoch {} has accuracy: {}".format(j/self.batch_size, i , train_acc))
            avg_cost /= n_batches
            print("Epoch {} has cost: {}".format(i, avg_cost))

