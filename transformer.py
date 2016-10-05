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
import sklearn
from eden import graph
from eden.util.display import draw_graph
import itertools
from sknn.mlp import Regressor, Layer
from sklearn import cross_validation
from eden.util import describe
from sklearn.linear_model import SGDClassifier
from eden.converter.rna.rnaplfold import rnaplfold_to_eden
from eden.converter.rna.rnafold import rnafold_to_eden
from sklearn.preprocessing import StandardScaler
import joblib
import time
from DeepNeuralNetwork import DeepNeuralNetwork
from eden.util import configure_logging

import logging
logger = logging.getLogger(__name__)


class Seq_to_Struct_Transform(object):

    def __init__(self, no_of_bits=10, struct_pre_processor=None, struct_vectorizer=None, d_seq=17, d_struct=1, radius=3,
                 min_rad=2, normalization=False, inner_normalization=True, nbits_seq=11, nbits_struct=11, random_state=1):

        self.pre_processor = struct_pre_processor        
        self.vectorizer = struct_vectorizer
        self.no_of_bits = no_of_bits           

        # parameters for vectorizer        
        self.params = dict()               
        self.params['d_seq']=d_seq
        self.params['d_struct']=d_struct
        self.params['min_r']=min_rad
        self.params['radius']=radius
        self.params['normalization']=normalization
        self.params['inner_normalization']=inner_normalization
        self.params['nbits_seq']=no_of_bits
        self.params['nbits_struct']=no_of_bits 

    def set_params(self, random_state=None):

        random.seed(random_state)
        
        # parameters for vectorizer                
        self.params['radius']=random.randrange(2,5,1)
        self.params['d_seq']=random.randrange(15,20,1)
        self.params['d_struct']=0
        self.params['min_r']=random.randrange(2,3,1) 
        self.params['normalization']=False
        self.params['inner_normalization']=True
        self.params['nbits_seq']=self.no_of_bits
        self.params['nbits_struct']=self.no_of_bits
		
		# Set the vectorizer
        self.vectorizer = graph.Vectorizer(r=self.params['radius'], d=self.params['d_struct'], min_r=self.params['min_r'], normalization=self.params['normalization'], 
                                                inner_normalization=self.params['inner_normalization'], nbits=self.params['nbits_struct'])
        return self.params

    def softmax(self, struct_std_data_matrx=None):

        maxes = np.amax(struct_std_data_matrx, axis=0)        
        #maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(struct_std_data_matrx - maxes)
        soft_struct_data_matrx = e / np.sum(e, axis=0)

        return soft_struct_data_matrx	  
    
    def seq_to_struct(self, sequences=None): 

        """
        Function to construct structure data matrix

        Parameters
        ----------

        sequences: list (default value = None):
            the list of sequences generated      
        struct_vectorizer (default value = None):
             converts to matrix

        Returns
        --------

        struct_data_matrx: 
            data matrices for sequences
        """
        params = self.params
        
        # Define the graphs
        graphs = mp_pre_process(sequences,
                                pre_processor=self.pre_processor,
                                pre_processor_args={},
                                n_jobs=-1)
                
        # Transform sequences to matrix
        struct_data_matrix = vectorize(graphs, vectorizer=self.vectorizer, n_jobs=-1)	

        # Densify the matrix
        struct_std_data_matrx = struct_data_matrix.toarray()            

        # Apply softmax to the matrix
        soft_struct_data_matrx = self.softmax(struct_std_data_matrx)

        return soft_struct_data_matrx

