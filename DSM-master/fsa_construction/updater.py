import os,sys
from fsa_construction.Standard_Automata import StandardAutomata
import fsa_construction.Standard_Automata as graph_lib

def parse_fsm_file(fsm_file,prefix4states=''):
    with open(fsm_file,'r') as reader:
        lines=[l.strip() for l in reader]

    index = 0
    
    num_startings = int(lines[index])
    index+=1

    startings = []
    for _ in range(num_startings):
        startings+=[prefix4states+lines[index]]
        index+=1
    
    num_endings = int(lines[index])
    index+=1

    endings = []
    for _ in range(num_endings):
        endings+=[prefix4states+lines[index]]
        index+=1
    
    num_edges = int(lines[index])
    index+=1


    edges=[]
    for  _ in range(num_edges):
        source,dest,label = lines[index].split()
        index+=1

        edges+=[(prefix4states+source,prefix4states+dest,label)]

    return StandardAutomata(startings,edges,endings)

def merge_fsms(Apath, Bpath):
    fsm_A = parse_fsm_file(Apath)
    fsm_B = parse_fsm_file(Bpath,prefix4states='BFSM_')

    fsm_C = fsm_A.clone()

    if len(fsm_A.startings) != 1 and len(fsm_B.startings) != 1:
        print("ERROR: this implementation can only merge two automata with single starting states!")
        sys.exit(-1)
    if len(fsm_A.endings) != 1 and len(fsm_B.endings) != 1:
        print("ERROR: this implementation can only merge two automata with single endingn states!")
        sys.exit(-1)

    united_starting = list(fsm_C.startings)[0]
    united_ending = list(fsm_C.endings)[0]
    for (source,dest,label) in fsm_B.transitions:
        if source in fsm_B.startings:
            source = united_starting
        
        if source in fsm_B.endings:
            source = united_ending

        if dest in fsm_B.startings:
            dest = united_starting
        
        if dest in fsm_B.endings:
            dest = united_ending
        fsm_C.transitions.add((source,dest,label))


    return fsm_C
def update(input_option):
    old_fsm_path = input_option.args.old_fsm

    local_fsm_path = input_option.args.work_dir+'/FINAL_mindfa.txt'

    updated_fsm_path = input_option.args.work_dir+'/UPDATED_mindfa.txt'

    updated_dot_path = input_option.args.work_dir+'/UPDATED_mindfa.dot'

    # merge two FSMs

    updated_fsm = merge_fsms(old_fsm_path,local_fsm_path)

    # minimize them

    merge_fsm = graph_lib.minimize_dfa(updated_fsm.nfa2dfa())


    if not os.path.isdir(os.path.dirname(updated_fsm_path)):
        os.makedirs(os.path.dirname(updated_fsm_path))

    with open(updated_fsm_path,'w') as writer:
        writer.write(merge_fsm.to_string()+'\n')

    merge_fsm.to_dot(updated_dot_path)

    return updated_fsm