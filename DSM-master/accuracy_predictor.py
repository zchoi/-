import os,sys
import argparse,multiprocessing
import train as RNNLM_training
import fsa_construction.input_processing as input_sampler
import fsa_construction.k_ptails as feature_extractor
import fsa_construction.clustering_pro as clustering_processing
import fsa_construction.estimate_accuracy as model_selection
import fsa_construction.updater as model_updater
import fsa_construction.Standard_Automata as graph_lib
from fsa_construction.Standard_Automata import StandardAutomata
import fsa_construction.clustering_pro as clustering_pro

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', type=str, default='input.txt',
                       help='path to file containing execution traces')
    
    parser.add_argument('--fsm', type=str, default='mindfa.txt',
                       help='directory to output a FSA')

    parser.add_argument('--out', type=str, default=None,
                       help='Path to output file containing prediction results')

    parser.add_argument('--verbose', action='store_true',
                       help='Path to output file containing prediction results')

   
    args = parser.parse_args()
    return args 

def predict(the_fsm,traces,verbose=False):
    fsm_pairs = set()
    adjlst = the_fsm.create_adjacent_list()
    for a in adjlst:
        for (b, label_one) in adjlst[a]:
            if b not in adjlst:
                continue
            for (c, label_two) in adjlst[b]:
                fsm_pairs.add((label_one, label_two))

    training_pairs = set()
    
    for tr in traces:
        training_pairs |= set([(tr[i], tr[i + 1]) for i in range(len(tr) - 1)])
    unseen_pairs = fsm_pairs - training_pairs
    predicted_precision = float(len(training_pairs)) / float(len(training_pairs | fsm_pairs))
    ######################
    accepted_traces_count = clustering_pro.count_accepted_traces(the_fsm,traces)
    predicted_recall = accepted_traces_count / len(traces)
    ######################
    if predicted_precision + predicted_recall != 0:
        fmeasure =  2.0 * predicted_precision * predicted_recall / (predicted_precision + predicted_recall)
    else:
        fmeasure=0.0

    if verbose:
        print("Unseen pairs:",unseen_pairs)
    

    return predicted_precision,predicted_recall,fmeasure

if __name__ == '__main__':
    args = read_args()
    
    the_fsm = graph_lib.parse_fsm_file(args.fsm)

    #######

    training_pairs = set()
    with open(args.traces, 'r') as reader:
        traces = [l.strip().split() for l in reader]
        traces = list(map(lambda x:x[1:] if x[0]=='<START>'else x,traces))
    
    precision,recall,fmeasure = predict(the_fsm,traces,verbose=args.verbose)

    print('Precision:',precision)
    print('Recall:',recall)
    print('F-measure:',fmeasure)

    if args.out is not None:
        args.out = os.path.abspath(args.out)
        if not os.path.isdir(os.path.dirname(args.out)):
            os.makedirs(os.path.dirname(args.out))
        with open(args.out,'w') as writer:
            writer.write('Precision:\t'+str(precision)+'\n')
            writer.write('Recall:\t'+str(recall)+'\n')
            writer.write('F-measure:\t'+str(fmeasure)+'\n')

