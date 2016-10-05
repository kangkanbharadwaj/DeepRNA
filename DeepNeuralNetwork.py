from eden.converter.rna.rnafold import rnafold_to_eden
from eden.converter.fasta import sequence_to_eden
from eden.converter.rna.rnaplfold import rnaplfold_to_eden
from eden.util import mp_pre_process
from eden.util import vectorize
from eden.graph import Vectorizer
from eden.util import vectorize
from eden import graph
import numpy as np
from sklearn import metrics
import random
from eden import graph
from eden.util.display import draw_graph
import itertools
from sknn.mlp import Regressor, Layer
from eden.util import describe
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import time
import copy

import logging
logger = logging.getLogger(__name__)

class DeepNeuralNetwork(object):

    def __init__(self, params=None, seq_pre_processor=None): 	              
                 
        self.scale = StandardScaler()                          
        self.pre_processor = seq_pre_processor
        self.params = params      
        
        if params != None:
            # Initialize the network
            self.net = Regressor(layers=params['layers'], learning_rate=params['learning_rate'], n_iter=params['n_iter'], dropout_rate=params['dropout_rate'],
                                     batch_size=params['batch_size'], regularize=params['regularize'], valid_size=params['valid_size'])

            # Initialize the vectorizer
            self.vectorizer = graph.Vectorizer(r=params['radius'], d=params['d_seq'], min_r=params['min_r'], normalization=params['normalization'], 
                                                    inner_normalization=params['inner_normalization'], nbits=params['nbits_seq'])  
                                 

    # Save the neural network object to the model_name path
    def save(self, model_name=None):        

        joblib.dump(self, model_name , compress=1)

    # Loads the neural network object from its respective path
    def load(self, model_name=None):

        self.__dict__.update(joblib.load(model_name).__dict__)  

    # Converts sequences to matrix
    def seq_to_data_matrix(self, sequences=None):               
                
        # Transform sequences to matrix
        graphs = mp_pre_process(sequences, pre_processor=self.pre_processor, pre_processor_args={}, n_jobs=-1)	        
                      
        seq_data_matrix = vectorize(graphs, vectorizer=self.vectorizer, n_jobs=-1)                

        # Densify the matrix
        seq_data_matrx = seq_data_matrix.toarray()

        # Standardize the matrix
        self.scale.fit(seq_data_matrx)
        std_seq_data_matrx = self.scale.transform(seq_data_matrx)        

        return std_seq_data_matrx

    # Training the network using traing sequences and the train structure matrix
    def fit(self, sequences=None, X_struct_std_train=None):

        # Convert sequences to data matrix
        X_seq_std_train = self.seq_to_data_matrix(sequences)    

        # Train the network
        self.net.fit(X_seq_std_train, X_struct_std_train)        

    # Predict the structure data matrix using testing sequences
    def predict(self, sequences=None):        

        # Convert sequences to data matrix
        X_seq_std_test = self.seq_to_data_matrix(sequences)
        
        # Predict the output matrix
        pred_data_matrix_out = self.net.predict(X_seq_std_test)             

        return pred_data_matrix_out                     

    # Function to train the network and predict the testing data
    def fit_predict(self, sequences_train=None, sequences_test=None, struct_matrix_train=None):                                               

        # Training the network using training sequences and the train structure matrix
        self.fit(sequences_train, struct_matrix_train)

        #Transform seq features to struct features using testing sequences
        return self.predict(sequences_test)                        
