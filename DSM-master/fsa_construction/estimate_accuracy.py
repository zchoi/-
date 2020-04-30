import sys, os


import lib
import fsa_construction.Standard_Automata as graph_lib
from fsa_construction.Standard_Automata import StandardAutomata
import random
import multiprocessing
from collections import Counter


def predict_accuracy(fsm_file, stat_file, input_file, prediction_file):
    print(fsm_file)
    num_cluster = int(os.path.basename(os.path.dirname(fsm_file)).replace('S_',''))
    training_pairs = set()
    with open(input_file, 'r') as reader:
        lines = [l.strip().split() for l in reader]
        for tr in lines:
            training_pairs |= set([(tr[i], tr[i + 1]) for i in range(len(tr) - 1)])
    ######################################
    fsm_pairs = set()
    the_fsm = graph_lib.parse_fsm_file(fsm_file)
    adjlst = the_fsm.create_adjacent_list()
    for a in adjlst:
        for (b, label_one) in adjlst[a]:
            if b not in adjlst:
                continue
            for (c, label_two) in adjlst[b]:
                fsm_pairs.add((label_one, label_two))
    ######################################
    predicted_precision = float(len(training_pairs)) / float(len(training_pairs | fsm_pairs))
    print("Predicted Precision:", predicted_precision, "unseen pairs:", len(
        fsm_pairs - training_pairs), "training pairs:", len(training_pairs))
    ######################################
    with open(stat_file, 'r') as reader:
        lines = [l.strip().split(':') for l in reader]
        recall =list( filter(lambda x: x[0].strip() == 'recall', lines))[0][-1].strip()
        recall=float(recall)
    print("Predicted Recall:", recall)
    ######################################
    if predicted_precision + recall > 0.0:
        predicted_f1 = float(2.0 * predicted_precision * recall) / float(predicted_precision + recall)
    else:
        predicted_f1 = 0.0
    print("Predicted F-measure",predicted_f1)
    with open(prediction_file, 'w') as writer:
        writer.write(str(predicted_precision) + '\n')
        writer.write(str(recall) + '\n')
        writer.write(str(predicted_f1) + '\n')
    return (fsm_file,num_cluster,predicted_precision,recall,predicted_f1)

def write_results(rankings,output_file):
    with open(output_file,'w') as writer:
        writer.write('ClusterPath,NumberOfClustersSetting,Precision,Recall,F-measure\n')
        for (fsm,num,p,r,f1) in rankings:
            cluster = os.path.dirname(fsm)
            writer.write(cluster+','+str(num)+','+str(p)+','+str(r)+','+str(f1)+'\n')

def extract_rejected_traces(uncovered_traces_file,rejected_traces_file):
    if not os.path.isfile(uncovered_traces_file):
        return
    with open(uncovered_traces_file,'r') as reader:
        lines = [l.strip() for l in reader]
    ans=[]
    g=[]
    for l in lines:
        if l.startswith('###################################'):
            if len(g)>0:
                ans+=[g[0].split()]
            g=[]
        else:
            g+=[l]
    if len(g)>0:
        ans+=[g[0].split()]
    if len(ans)>0:
        if not os.path.isdir(os.path.dirname(rejected_traces_file)):
            os.makedirs(os.path.dirname(rejected_traces_file))
        with open(rejected_traces_file,'w') as writer:
            for tr in ans:
                writer.write('<START> '+ ' '.join(tr)+'\n')
        
def selecting_model(input_options):

    cluster_folders = lib.find_folders_by_prefix(input_options.clustering_space_dir, 'S_')
    rankings=[]

    pool = multiprocessing.Pool(processes=input_options.args.max_cpu)
    paras=[]    

    for single_cluster_folder in cluster_folders:
        fsm_file = single_cluster_folder + '/fsm.txt'
        if not os.path.isfile(fsm_file):
            print ("ERROR: cannot find",fsm_file)
            continue
        stat_file = single_cluster_folder + '/statistic.txt'
        prediction_file = single_cluster_folder + '/accuracy_prediction.txt'
        a=(fsm_file, stat_file, input_options.raw_input_trace_file, prediction_file)
        paras+=[a]
        #r=predict_accuracy(fsm_file, stat_file, input_options.raw_input_trace_file, prediction_file)
    rankings=pool.starmap(predict_accuracy,paras)
    pool.close()
    pool.join()

    rankings.sort(key=lambda x:(x[-1],x[1]),reverse=True)
    best_fsm_folder =os.path.dirname(rankings[0][0])
    write_results(rankings,input_options.args.work_dir+'/model_selections.csv')
    import shutil
    for name in ['mindfa','dfa','fsm']:
        if not os.path.isfile(best_fsm_folder+'/'+name+'.txt'):
            continue
        shutil.copyfile(best_fsm_folder+'/'+name+'.txt',input_options.args.work_dir+'/FINAL_'+name+'.txt')
        shutil.copyfile(best_fsm_folder+'/'+name+'.eps',input_options.args.work_dir+'/FINAL_'+name+'.eps')

        extract_rejected_traces(best_fsm_folder+'/dfa_uncovered_traces.txt',input_options.args.work_dir+'/rejected_traces/input.txt')
        return input_options.args.work_dir+'/FINAL_'+name+'.txt'
        
