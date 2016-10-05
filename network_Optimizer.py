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
import sys

sys.path.append('/home/bharadwk/DeepNet/basic/')

from DeepNeuralNetwork import DeepNeuralNetwork
from transformer import Seq_to_Struct_Transform

import logging
logger = logging.getLogger(__name__)
from eden.util import serialize_dict

#from eden.util import configure_logging
#configure_logging(logger,verbosity=0)
#fhandler = logger.FileHandler(filename="/home/bharadwk/DeepRNA/log_file/mylog.log", mode='w')
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fhandler.setFormatter(formatter)
#logger.addHandler(fhandler)
#logger.setLevel(logging.DEBUG)

class NetworkOptimizer(object):

    def __init__(self, no_of_bits=11, random_state=1, net=None, n_features_in=1, n_features_out=23, n_features_hidden=23, regularize_type='L1', 
                 output_layer_type='Softmax', hidden_layer_type='Rectifier', n_layers=3, learn_rate=0.001, no_of_iter=100, 
                 batch_size=10, valid_size=0.1, seq_scale=None, seq_pre_processor=None, struct_pre_processor=None, dropout_rate=0.2, 
                 seq_vectorizer=None, struct_vectorizer=None, layers=None, transformer=None):

        self.no_of_bits = no_of_bits
        self.seq_pre_processor = seq_pre_processor
        self.struct_pre_processor = struct_pre_processor
        self.seq_vectorizer = seq_vectorizer
        self.struct_vectorizer = struct_vectorizer
        self.random_state = random_state         
        self.net = net
        self.seq_scale = seq_scale        
        self.n_features_hidden = n_features_hidden
        self.n_features_out = n_features_out
        self.output_layer_type = output_layer_type
        self.hidden_layer_type = hidden_layer_type
        self.n_layers = n_layers
        self.layers=[]
        for i in range(self.n_layers):
            self.layers.append(Layer(self.hidden_layer_type, units=self.n_features_hidden, name='hidden%d'%i))
        self.layers.append(Layer(self.output_layer_type, units=self.n_features_out))

        self.params = dict()        
        # parameters for neural network
        self.params['n_features_in'] = n_features_in 
        self.params['n_features_out'] = self.n_features_out
        self.params['n_features_hidden'] = self.n_features_hidden
        self.params['layers']=layers       
        self.params['learning_rate']=learn_rate
        self.params['n_iter']=no_of_iter
        self.params['batch_size']=batch_size
        self.params['regularize']=regularize_type
        self.params['valid_size']=0.1                
        self.params['random seed value']=random_state
        self.params['dropout_rate']=dropout_rate    
        
        # Initialize class transform
        self.transformer=transformer                    

    def randomize(self, random_state=1):

        """    
        Function to generate the parameters for Regressor neural network and vectorizer randomly 

        Parameters
        ----------

        no_of_bits: int (default value = None)
            number of bits
        random_state: int (default value = 1)
            variable to define the seed random seed value

        Variables
        ---------

        n_features_in: 
            input features in terms of number of neurons for input layer
        n_features_out: 
            output features in terms of number of neurons for output layer
        n_features_hidden: 
            hidden features in terms of number of neurons for hidden layer
        layers: 
            a list of hidden and output layers
        list_of_hidden_layers: 
            defines the differnet types(can be linear or non-linear) of hidden layers
        list_of_output_layers: 
            defines the differnet types(can be linear or non-linear) of output layers
        r: 
            The maximal radius size. 
        d_seq: 
            The maximal distance size of sequence
        d_struct: 
            The maximal distance size of structure
        min_r: 
            The minimal radius size.
        learning_rate: 
            Real number indicating the default/starting rate of adjustment for the weights 
            during gradient descent
        n_iter: 
            The number of iterations of gradient descent to perform on the neural networks weights when training with fit()
        batch_size: 
            Number of training samples to group together when performing stochastic gradient descent
        regularize: 
            Technique in machine learning to prevent overfitting. 
            Mathematically L2 is sum of the square of the weights, while L1 is sum of the weights.
        valid_size: 
            Ratio of the training data to be used for validation. 
        normalization: 
            If set the resulting feature vector will have unit euclidean norm.
        inner_normalization: 
            If set the feature vector for a specific combination of the radius and distance size
            will have unit euclidean norm.
        nbits_seq: 
            The number of bits that defines the feature space size for sequence
        nbits_struct: 
            The number of bits that defines the feature space size for structure

        Returns:
        --------
        A dictionary of parameters to be used by neural network and vectorizer 
        """

        random.seed(random_state)

        n_features_in = 2**self.no_of_bits+1
        self.n_features_out = 2**self.no_of_bits+1    
        #self.n_features_hidden = max(n_features_in, self.n_features_out) * 3
        self.n_features_hidden = random.randrange(100,10000)    

        self.layers=[]
        list_of_hidden_layers = ['Rectifier','Sigmoid','Tanh']
        list_of_output_layers = ['Linear','Softmax']
        list_of_regularize = ['L1','L2']
        regularize_type = random.choice(list_of_regularize)
        output_layer_type = random.choice(list_of_output_layers)
        hidden_layer_type = random.choice(list_of_hidden_layers)

        self.n_layers=random.randrange(1,4)    
        for i in range(self.n_layers):
            self.layers.append(Layer(hidden_layer_type, units=self.n_features_hidden, name='hidden%d'%i))
        self.layers.append(Layer(output_layer_type, units=self.n_features_out))             
                      
        # parameters for neural network        
        self.params['n_features_in'] = n_features_in 
        self.params['n_features_out'] = self.n_features_out
        self.params['n_features_hidden'] = self.n_features_hidden
        self.params['layers']=self.layers        
        self.params['learning_rate']=0.001
        self.params['n_iter']=random.randrange(100,150,2)
        self.params['batch_size']=10
        self.params['regularize']=regularize_type
        self.params['valid_size']=0.1        
        self.params['random seed value']=random_state  
        self.params['dropout_rate']=random.uniform(0.1,0.8)                         

        return self.params              

    def estimate_data_representation_equivalence(self, X_struct_std_test=None, X_struct_std_pred_test=None):	

        """
        Function to validate the original data matrix and predicted data matrix using cross validation 

        Parameters
        ----------

        parameters_dictionary: dict (default value = None)
            dictionary of parameters to be used by Regressor neural network and vectorizer
        sequences: list (default value = None)
            the list of sequences generated 

        Returns
        --------

        ROC_mean_score, ROC_std_dev_score: 
            Mean ROC score with Standard Deviation after validation    
        """             		      

        X = np.vstack((X_struct_std_test,X_struct_std_pred_test))
        y = np.array([1]*X_struct_std_test.shape[0] + [-1]*X_struct_std_pred_test.shape[0])      

        predictor = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=-1)		
        scores = cross_validation.cross_val_score(predictor, X, y, cv=10, scoring='roc_auc')    
        ROC_mean_score = np.mean(scores)
        ROC_std_dev_score = np.std(scores)        
        print('AUC ROC: %.4f +- %.4f' % (ROC_mean_score,ROC_std_dev_score))         
        return ROC_mean_score, ROC_std_dev_score


    def optimize(self, sequences=None, init_params=1):

        """
        Function to generate a list of average of ROC mean and ROC standard deviation and its 
        corresponding parameters 

        Parameters
        ----------

        sequences: list (default value = None)
            the list of sequences generated      
        no_of_times_of_parameter_initialization: int (default value = 1)
            number of times we call parameter generation function to randomly initialize the parameters
        no_of_times_fit_predict: int (default value = 1)
            number of times for a set of parameters we fit()/predict() the samples using neural network

        Returns
        --------

        list_of_ROC_score_and_parameters: 
            a list of average of ROC mean and ROC standard deviation and corresponding parameters
        """

        opt_net = None
        min_mean_ROC = 0.0
        min_std_dec_ROC = 0.0
        params_to_log = dict()
        
        # Split sequences to train and test
        sequences_train = sequences[:len(sequences)/2]
        sequences_test = sequences[len(sequences)/2:]

        # Get training and testing structure matices
        struct_matrix_train = self.transformer.seq_to_struct(sequences_train)
        struct_matrix_test = self.transformer.seq_to_struct(sequences_test)                  
            
        min_score=1        

        for i in range(init_params):

            # Set the network parameters
            params=self.randomize(self.random_state)                                            

            # Concetenate the parameter dictionaries
            parameters = dict(params.items() + self.transformer.params.items()) 	                                                               		
            
            # instatiate neural network class
            deep_neural_network = DeepNeuralNetwork(params=parameters, seq_pre_processor=self.seq_pre_processor)
            
            # Train the model and get the predicted matrix
            predict_matrix = deep_neural_network.fit_predict(sequences_train, sequences_test, struct_matrix_train)

            # Compare the predicted to the original matrix
            ROC_mean_score, ROC_std_dev_score = self.estimate_data_representation_equivalence(struct_matrix_test, predict_matrix)
            curr_score = abs(float(ROC_mean_score) - 0.5) + ROC_std_dev_score
            if  curr_score < min_score:
                min_mean_ROC = ROC_mean_score
                min_std_dev_ROC = ROC_std_dev_score
                params_to_log = parameters			                   
                min_score = curr_score                
                opt_net = deep_neural_network
                #deep_neural_network.save()        
        
        logger.info('\n\n')
        logger.info('On train set:')
        logger.info('AUC ROC: %.4f +- %.4f' % (min_mean_ROC,min_std_dev_ROC))         
        logger.info('\n\n')
        logger.info('Trained and tested with parameters:\n\n %s \n\n' % serialize_dict(params_to_log))          
        
        return opt_net
