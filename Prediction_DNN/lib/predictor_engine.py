# Import the necessary libraries.
import pandas as pd
import numpy as np
#from time import time
import tensorflow as tf
import itertools
from time import time
from datetime import datetime

from core_functions import *


class Predictor(object):
    def __init__(self, din = {}):
        self.dict = {
            'out_classifier_filename': './models/football_classifier_serieA_DNN_extended',
            'batch_size' : 200,
            'X_train' : None,
            'y_train' : None,
            'X_valid' : None,
            'y_valid' : None,
            'X_test'  : None,
            'y_test'  : None,
        }
        self.dict.update(din)
    

    
        # The training set is used to train the network
        self.train_dataset          = self.dict['X_train'].values.tolist()
        self.train_labels           = self.dict['y_train'].values.tolist()
        self.train_dataset_batch    = [] # remember: a batch is a subset of the entire dataset on which I do the training
        self.train_labels_batch     = []
        self.train_predictions      = []
        self.train_predDecoded      = []
        self.train_accuracy         = None
        self.train_accuracies       = []
        
        # The validation set is used to calibrate the network hyperparameters
        self.valid_dataset          = self.dict['X_valid'].values.tolist()
        self.valid_labels           = self.dict['y_valid'].values.tolist()
        self.valid_predictions      = []
        self.valid_predDecoded      = []
        self.valid_accuracy         = None # validation accuracies obtaned for different regularizations
        self.valid_accuracies       = []
        
        # The test set is used to test the network on new data with unknown labels
        self.test_dataset           = self.dict['X_test'].values.tolist()
        self.test_labels            = self.dict['y_test'].values.tolist()
        self.test_predictions       = [] # predictions done on the test set
        # there is no test_accuracy since it is assumed that no a-priori info is available on the test data
        
        self.ntrain, self.nfeatures = self.dict['X_train'].shape
        _          , self.nlabels   = self.dict['y_train'].shape
        self.nvalid, _              = self.dict['X_valid'].shape
        self.ntest , _              = self.dict['X_test'].shape
        
        print("[PREDICTOR] Created predictor: training %d, validation %d, test %d, [%d features and %d labels]"%(self.ntrain, self.nvalid, self.ntest, self.nfeatures, self.nlabels))
        
        
    def train(self,
            num_steps = 20001,
            regul_val = [1e-8,0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05], # must be a list, even with one element
            verbose   = True,
            verbosity = 0.1 # statistics displayed every (num_steps*verbosity) iterations
        ):
        
        self.valid_accuracy = 0
        # Model training  
        #self.valid_accuracy
        print("[TRAIN] Initialized, tested beta_regul: ", regul_val)
        print('[TRAIN-START] ', datetime.now())
        print("Regul\t\tstep\tloss\tMinibatch acc.\tValid. acc.")
        for regul in regul_val: 
            with tf.Session(graph=self.graph) as session:
                tf.global_variables_initializer().run()
                writer = tf.summary.FileWriter("./tmp/graph.png", session.graph)
                
                step = 1        
                time_start = time()
                while True:
                    # Generate a minibatch.
                    train_dataset_batch, train_labels_batch = self.__next_batch__(self.dict['batch_size'], self.train_dataset, self.train_labels)
        
                    # Prepare a dictionary telling the session where to feed the minibatch.
                    feed_dict = {
                        self.tf_train_dataset : train_dataset_batch,
                        self.tf_train_labels  : train_labels_batch,
                        self.beta_regul       : regul}
                    # Run the session
                    _, loss, train_predictions = session.run( [self.optimizer, self.loss, self.tf_train_prediction], feed_dict=feed_dict )
                    valid_predictions = self.tf_valid_prediction.eval()
            
                    train_accuracy  = calculate_accuracy(train_predictions, train_labels_batch)
                    valid_accuracy  = calculate_accuracy(valid_predictions, self.valid_labels)
            
                    # Monitor the training
                    if verbose and (step % int(num_steps*verbosity) == 0 ): 
                        print("%f\t%d\t%.1f\t%.1f%%\t\t%.1f%%" % (regul, step, loss, train_accuracy, valid_accuracy))
                
                    ## Termination criteria
                    # 1) Stop the training if there is no convergence in an early stage
                    #if step > num_steps/3 and train_accuracy < 70:
                    #    print("%f\t%d\t%.1f\t%.1f%%\t\t%.1f%%" % (regul, step, loss, train_accuracy, valid_accuracy))
                    #    print('[TRAIN-STOP] Not converging')
                    #    break
            
                    if step == num_steps:
                        print("%f\t%d\t%.1f\t%.1f%%\t\t%.1f%%" % (regul, step, loss, train_accuracy, valid_accuracy))
                        print('[TRAIN-STOP] Max iter reached')
                        break
                    step += 1   
         
                print('[TRAIN-END] Elapsed', np.round(time()-time_start, 1) , "secs")
                
                self.train_accuracies.append( train_accuracy )
                self.valid_accuracies.append( valid_accuracy )
                writer.close()
                
                # Assess which regul is providing the best results and store the classifier
                if valid_accuracy > self.valid_accuracy:
                    self.saver.save(session, self.dict['out_classifier_filename'])
                    print('[TRAIN] Saved classifier with regul: %f'% (regul))
                     
                    # Get prediction in a more readable form
                    self.train_dataset_batch = train_dataset_batch
                    self.train_labels_batch  = train_labels_batch
                    self.train_predictions   = train_predictions
                    self.valid_predictions   = valid_predictions
                    self.train_accuracy      = train_accuracy
                    self.valid_accuracy      = valid_accuracy           
                
                    # Once the optimal network has been found, it is possible it is tested
                    self.test_predictions = self.predict(session, self.test_dataset)
                
        return
                        
                        
        
    def predict(self, session, X_dataset):
        predictions = []
        for x in X_dataset:
            feed_dict = {self.tf_test_dataset : [x]}
            predictions.append( list( session.run( [self.tf_test_prediction], feed_dict=feed_dict)[0][0]) )
        return predictions
                    
    
    def build_model(self, hidden_layers = []):
        # layers is an array of dicts and contains a description of the network structure, e.g.:
        # hidden_layers = [
        #    {'nnodes':20, 'keep_prob':0.5}, # 1st hidden layer
        #    {'nnodes':10, 'keep_prob':0.5}, # 2nd hidden layer
        #    {'nnodes':5 , 'keep_prob':0.5}, # 3rd hidden layer
        # ]
                
        # add output layer to the config parameters. Note that the output layer does not have a dropout
        layers = [*hidden_layers, { 'nnodes':self.nlabels, 'keep_prob':1 }]
        print('[DESIGN] Created a network with %d layers'%(len(layers)+1) )
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            self.tf_train_dataset = tf.placeholder(tf.float32, shape=(self.dict['batch_size'], self.nfeatures))
            self.tf_train_labels  = tf.placeholder(tf.float32, shape=(self.dict['batch_size'], self.nlabels))
            self.tf_valid_dataset = tf.constant( self.valid_dataset )
            self.tf_test_dataset  = tf.placeholder(tf.float32, shape=(1, self.nfeatures), name='predict_dataset')
            global_step      = tf.Variable(0)
            self.beta_regul       = tf.placeholder(tf.float32)
    
            # Variables.
            ninputs  = self.nfeatures
            weights = []
            biases  = []
            for idx, pars in enumerate(layers):
                wght = tf.Variable( tf.truncated_normal([ninputs, pars['nnodes']], stddev=np.sqrt(2.0/ninputs )) )
                weights.append( wght )
                biases.append( tf.Variable(tf.zeros([pars['nnodes']])) )
                ninputs = pars['nnodes']

  
            # Build the network
            logits = None
            layer  = self.tf_train_dataset
            regularization_loss = tf.nn.l2_loss(weights[0])
            for idx in range(len(weights)-1):
                pars = layers[idx]
                wght = weights[idx]
                bias = biases[idx]
                
                layer = tf.nn.relu(tf.matmul(layer, wght) + bias)
                regularization_loss += tf.nn.l2_loss(weights[idx+1])
                
                if pars['keep_prob'] < 1: # do not perform dropout if keep_prob probability is 1
                    layer = tf.nn.dropout(layer, pars['keep_prob'])
            
            logits = tf.matmul(layer, weights[-1]) + biases[-1]
            
            # Training computation.
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=logits)) +\
                self.beta_regul * regularization_loss
  
            # Optimizer.
            #learning_rate = tf.train.exponential_decay(0.5, global_step, 4000, 0.65, staircase=True)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step)

            # Predictions for the training, validation, and test data.
            self.tf_train_prediction = tf.nn.softmax(logits)
            
            layer_valid = self.tf_valid_dataset
            layer_test  = self.tf_test_dataset
            for idx in range(len(weights)-1):
                pars = layers[idx]
                wght = weights[idx]
                bias = biases[idx]
                
                layer_valid = tf.nn.relu(tf.matmul(layer_valid, wght) + bias)
                layer_test  = tf.nn.relu(tf.matmul(layer_test, wght) + bias)
                
            self.tf_valid_prediction = tf.nn.softmax(tf.matmul(layer_valid, weights[-1]) + biases[-1])
            self.tf_test_prediction  = tf.nn.softmax(tf.matmul(layer_test, weights[-1]) + biases[-1], name='predict')
    
            #Create a saver object which will save all the variables
            self.saver = tf.train.Saver()
            
            
        
    def __next_batch__(self, num, data, labels):
        '''
        Return a total of `num` random samples and labels. 
        '''
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)