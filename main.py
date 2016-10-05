import sys
import os
import getopt
import argparse
import datetime

import logging
from eden.util import configure_logging
from eden.util import serialize_dict
from os.path import basename
import time

import random
import numpy as np
import matplotlib.pyplot as plt
from eden.converter.rna.rnafold import rnafold_to_eden
from eden.converter.fasta import sequence_to_eden
from eden.converter.rna.rnaplfold import rnaplfold_to_eden
from eden.graph import Vectorizer
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier

sys.path.append('/home/bharadwk/DeepNet/lib/')
from network_Optimizer import NetworkOptimizer

sys.path.append('/home/bharadwk/DeepNet/basic/')
from transformer import Seq_to_Struct_Transform

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def generate_rna_seq(gc_content, length):    
    n_g=int(gc_content * length / 2)
    n_c=int(gc_content * length / 2)
    n_a=int((1 - gc_content) * length / 2)
    n_u=int((1 - gc_content) * length / 2)

    seq = list('G'*n_g + 'C'*n_c + 'A'*n_a + 'U'*n_u)
    random.shuffle(seq)
    #print "sequence after shuffling" %seq
    seq = ''.join(seq)
    #print "sequence after joining" %seq
    return seq

def generate_rna_seqs(gc_content, length, num):
    seqs = []    
    for i in range(num):
        seqs.append(('ID_%d'%i, generate_rna_seq(gc_content, length)))
    return seqs

def pre_processor_rnaplfold(seqs):    
    graphs = rnaplfold_to_eden(seqs,
                               window_size = 250,
                               max_bp_span = 35,
                               avg_bp_prob_cutoff = 0.4,
                               max_num_edges = 1,
                               no_lonely_bps=True,
                               nesting=True)
    
    return graphs

'''RNAfold'''
def pre_processor_rnafold(seqs):
    graphs = rnafold_to_eden(seqs)
    
    #from eden.modifier.graph import structure 
    #graphs = structure.basepair_to_nesting(graphs)
    return graphs   


def argparse_setup():

    parser = argparse.ArgumentParser()
    parser.add_argument("num", type=int, help="specifies number of sequences")
    parser.add_argument("length", type=int, help="specifies length of a sequence")
    parser.add_argument("gc_content", type=float, help="specifies the G_C percentage in a sequence")
    parser.add_argument("nbits", type=int, help="specifies number of bits of a particular sequence")
    parser.add_argument("random_state", type=int, help="used for hardcoding a random number to be used as a random state for parameter initialization")
    parser.add_argument("init_params", type=int, help="with how many sets of parameters the model needs to learn")        
    parser.add_argument("-v", "--verbosity", action="count", help="Increase output verbosity")
    parser.add_argument("-x", "--no_logging", dest="no_logging", help="If set, do not log on file.", action="store_true")
    parser.add_argument("-o", "--output_dir", dest="output_dir_path", help="Path to output directory.", default="out", type=str)
    parser.add_argument("-l", "--logging_dir", help="Path to log directory.", default="out", type=str)
    parser.add_argument("-m", "--model_file", dest="model_file", help="Model file name. Note: it will be located in the output directory.", default="model", type=str)
    
    return parser

def main(args=None, logger=None):	     
            
    #data creation with default values
    num=50
    length=25
    gc_content=.6
    nbits=10
    random_state=87    
    init_params=1                
        
    seqs = generate_rna_seqs(gc_content=args.gc_content, length=args.length, num=args.num)
    
    transformer = Seq_to_Struct_Transform(no_of_bits=args.nbits, struct_pre_processor=pre_processor_rnafold)
    transformer.set_params(args.random_state)
        
    #init optimizer
    network_optimizer = NetworkOptimizer(no_of_bits=args.nbits, random_state=args.random_state, seq_pre_processor=sequence_to_eden, transformer=transformer)

    deep_neural_network = network_optimizer.optimize(sequences=seqs, init_params=args.init_params)                                                                                
        
    out_dir= args.output_dir_path + 'network_model_gc%.2f_len%d_num%d/'%(args.gc_content,args.length,args.num)        
    _dir=os.path.dirname(out_dir)
    
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    
    mod_fname = _dir+'/net.model'
    
    if os.path.exists(mod_fname):
        print "file exists"
    else:
        open(mod_fname, 'w')           
    
    deep_neural_network.save(model_name = mod_fname)    

    test_seqs = generate_rna_seqs(gc_content=gc_content, length=length, num=num)
           
    struct_data_matrx = transformer.seq_to_struct(test_seqs)                  

    #load optimized model
    from DeepNeuralNetwork import DeepNeuralNetwork
    test_deep_neural_network = DeepNeuralNetwork()  
   
    test_deep_neural_network.load(model_name = mod_fname)
    pred_data_matrix_out = test_deep_neural_network.predict(test_seqs)       

    #estimate quality
    X = np.vstack((struct_data_matrx,pred_data_matrix_out))
    y = np.array([1]*struct_data_matrx.shape[0] + [-1]*pred_data_matrix_out.shape[0])

    predictor = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=-1)
    scores = cross_validation.cross_val_score(predictor, X, y, cv=10, scoring='roc_auc')

    ROC_mean_score = np.mean(scores)
    ROC_std_dev_score = np.std(scores)

    logger.info('\n\n')
    logger.info('On test set:')
    logger.info('AUC ROC: %.4f +- %.4f' % (ROC_mean_score,ROC_std_dev_score))    
    logger.info('\n\n')       

def main_script(prog_name=None, logger=None):
    
    parser = argparse_setup()
    args = parser.parse_args()

    if args.no_logging:
        configure_logging(logger, verbosity=args.verbosity)
    else:
		configure_logging(logger, verbosity=args.verbosity, filename=args.logging_dir + 'logs_gc%.2f_len%d_num%d'%(args.gc_content,args.length,args.num) + '.log')

    logger.debug('-' * 80)
    logger.debug('Program: %s' % prog_name)
    logger.debug('\n')
    logger.debug('Called with parameters:\n\n %s \n\n' % serialize_dict(args.__dict__))
    
    start_time = time.asctime( time.localtime(time.time()) )
    logger.info('Initializing program execution %s \n\n' % (start_time))   
    try:
        main(args, logger)
    except Exception:
        import datetime
        curr_time = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        logger.exception("Program run failed on %s" % curr_time)
        exit(1)
    finally:
        end_time = time.asctime( time.localtime(time.time()) )        
        logger.info('Executing program execution %s' % (end_time)) 
        logger.info('-' * 80)                  

if __name__ == '__main__':
   main_script(prog_name=os.path.basename(__file__), logger=logging.getLogger()) 
