import os,sys
import argparse,multiprocessing
import train as RNNLM_training
import fsa_construction.input_processing as input_sampler
import fsa_construction.k_ptails as feature_extractor
import fsa_construction.clustering_pro as clustering_processing
import fsa_construction.estimate_accuracy as model_selection
import fsa_construction.updater as model_updater
import matplotlib.pyplot as plt
class Option:
    def __init__(self,args):
        self.update_mode=False
        if args.old_fsm is not None:
            if args.additional_trace is None:
                print("Please provide all information to update the automata")
                sys.exit(-1)
            
            self.update_mode=True
        ####################################################################################
        self.args = args
        if not os.path.isdir(self.args.work_dir):
            os.makedirs(self.args.work_dir)
        if self.args.additional_trace is None:
            self.raw_input_trace_file = self.args.data_dir+'/input.txt'
        else:
            self.raw_input_trace_file = self.args.additional_trace+'/input.txt'
            self.input_training_trace_file = self.args.data_dir+'/input.txt'
        #######
        if not os.path.isfile(self.raw_input_trace_file):
            print("Cannot find input execution traces stored in", self.raw_input_trace_file)
            sys.exit(-1)
        # self.preprocessed_trace_file = self.args.data_dir+'/input.txt'
        self.cluster_trace_file = self.args.data_dir+'/cluster_traces.txt'
        self.clustering_space_dir = self.args.work_dir+'/clustering_space'
        self.features4clustering_dir = self.args.work_dir+'/features4clustering'
        ####################################################################################

        self.generated_traces_folder = self.features4clustering_dir
        self.validation_traces_folder = self.raw_input_trace_file
        self.output_folder=self.clustering_space_dir
        self.min_cluster = self.args.min_cluster
        self.max_cluster= self.args.max_cluster
        self.seed = self.args.seed
        self.dfa = self.args.dfa
        self.dbscan_eps = self.args.dbscan_eps
        

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='input_traces',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--work_dir', type=str, default='work_dir',
                       help='directory to output a FSA')

    parser.add_argument('--max_cpu', type=int, default=4,
                       help='Maximum number of processors for parallel processing')

    ##### RNNLM Learning parameters ####

    parser.add_argument('--rnn_size', type=int, default=32,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='rnn',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10000,
                       help='save frequency')
    # parser.add_argument('--gpu_mem', type=float, default=0.666,
    #                     help='% of gpu memory to be allocated to this process. Default is 66.6%')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)

    #### parameters for updating an existing automaton ####

    parser.add_argument('--old_fsm', type=str, default=None,
                       help='Path to a file containing the existing automaton')
    parser.add_argument('--additional_trace', type=str, default=None,
                       help='Path to a folder containing input.txt that has additional traces for updating the existing automaton')

    #### clustering parameters ####

    parser.add_argument('--max_cluster', type=int, default=20,
                        help='Max. number of clusters setting')
    parser.add_argument('--min_cluster', type=int, default=2,
                        help='Min. number of clusters setting')
    parser.add_argument('--seed', type=int, default=9999,
                        help='Initialized seed')
    parser.add_argument('--dbscan_eps', type=float, default=0.1,
                        help='DBSCAN\'s eps parameter (for future release)')
    parser.add_argument('--dfa', type=int, default=1,
                        help='Create DFA')
    args = parser.parse_args()
    return Option(args) 


if __name__ == '__main__':
    input_option = read_args()

    ######## preprocessing traces & trace sampling ###########

    input_sampler.select_traces(input_option.raw_input_trace_file,input_option.cluster_trace_file)

    ######## train RNNLM model ########

    if not os.path.isdir(input_option.args.save_dir) or (input_option.args.additional_trace is not None and input_option.args.init_from is not None):
        if not os.path.isdir(input_option.args.save_dir):
            os.makedirs(input_option.args.save_dir)
        p = multiprocessing.Process(target=RNNLM_training.train, args=(input_option.args,))
        p.start()
        p.join()
        #l,index=RNNLM_training.train(input_option.args)
        #plt.plot(index,l)
        #plt.xlabel("iterations")
        #plt.ylabel("train_loss")
        #plt.show()

    ######## feature extraction ########

    feature_extractor.feature_engineering(input_option)

    ######## clustering ########

    clustering_processing.clustering_step(input_option)

    ######## model selection #######

    final_file=model_selection.selecting_model(input_option)

    print("Done! Final FSM is stored in",final_file)

    ######## merge two automata ######
    
    if input_option.update_mode:
        
        model_updater.update(input_option)
