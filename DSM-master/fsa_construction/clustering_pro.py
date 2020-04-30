import gzip
import math
import os
import sys

from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from fsa_construction.Standard_Automata import StandardAutomata
import fsa_construction.Standard_Automata as graph_lib

import argparse,multiprocessing
import lib


def max_trace_depth():
    return 10000


def waiting_time():
    return 60 * 5


def do_clustering(args, to_cluster_X, clustering_algorithm=None, ncluster=None, eps=None):
    if clustering_algorithm is None:
        clustering_algorithm = args.alg
    if ncluster is None:
        ncluster = args.num_cluster
    if eps is None:
        eps = args.dbscan_eps
    print("Clustering with", clustering_algorithm, ncluster, eps)
    if clustering_algorithm == 'kmeans':
        cluster_estimator = KMeans(n_clusters=ncluster, random_state=9999).fit(to_cluster_X)
    elif clustering_algorithm == 'hierarchical':
        cluster_estimator = AgglomerativeClustering(n_clusters=ncluster, linkage='ward').fit(to_cluster_X)
    elif clustering_algorithm == 'hierarchical_average':
        cluster_estimator = AgglomerativeClustering(n_clusters=ncluster, linkage='average').fit(to_cluster_X)
    elif clustering_algorithm == 'hierarchical_complete':
        cluster_estimator = AgglomerativeClustering(n_clusters=ncluster, linkage='complete').fit(to_cluster_X)
    elif clustering_algorithm == 'affinity_propagation':
        cluster_estimator = AffinityPropagation().fit(to_cluster_X)
    elif clustering_algorithm == 'dbscan':
        cluster_estimator = DBSCAN(eps=eps).fit(to_cluster_X)
    else:
        print("ERROR: unknown state_clustering algorithm", clustering_algorithm)
        sys.exit(0)
    return cluster_estimator


def parse_feature_vector(arr):
    ans = {}
    for e in arr:
        x = e.split(':')
        word = ':'.join(x[:-1])
        value = x[-1]
        ans[word] = value
    return ans


def parse_probs_lines(lines, method_list):
    two_tails = {}
    for fline in lines:
        fline = fline.split()
        if fline[0].startswith('1-TAIL'):
            one_tail = parse_feature_vector(fline[1:])
        elif fline[0].startswith('2-TAIL'):
            next_method = fline[1]
            two_tails[next_method] = parse_feature_vector(fline[2:])
        else:
            print("ERROR: unrecognizable lines:", fline)
            sys.exit(0)

    vt = map(lambda x: math.log10(float(one_tail[x])), method_list)

    # to include two tails features
    # for m in method_list:
    #     vt += map(lambda x: two_tails[m][x], method_list)

    return tuple(vt)


def parse_validation_traces(validation_traces_folder, prefix_name):
    traces_files = lib.find_files_by_prefix(validation_traces_folder, prefix_name)
    ans = []
    for tr_file in traces_files:
        with open(tr_file, 'r') as reader:
            lines = [l.strip() for l in reader]
            words =list( map(lambda x: x.split()[-1], filter(lambda x: x.startswith('WORD'), lines)))
            while words[0] == starting_char():
                words = words[1:]
            if words[-1] != ending_char():
                words += [ending_char()]
            ans += [tuple(words)]
    return set(ans)


def find_ending_methods(possible_ending_words, lines):
    the_trace = [lines[i].split()[-1] for i in range(len(lines)) if lines[i].startswith('WORD')]

    if the_trace[-1] == lib.ending_char():
        possible_ending_words.add(the_trace[-2])
    else:
        print("ERROR: last method is not <END>")
        print(the_trace)
        sys.exit(0)


def is_feature_file(name, prefix):
    name = name.replace(prefix, '')
    name = name[:name.find('.')]
    try:
        int(name)
        return True
    except:
        return False


def parse_sampled_traces(generated_traces_folder, prefix_name):
    seed_traces_files = lib.find_files_by_prefix(generated_traces_folder, prefix_name)
    # print seed_traces_files,generated_traces_folder
    print("Found", len(seed_traces_files), "training trace files")
    trace_set = set()
    method_set = set([lib.starting_char(), lib.ending_char()])
    validation_traces = []
    possible_ending_words = set()
    for f in seed_traces_files:
        print("Processing", f)
        if not is_feature_file(os.path.basename(f), prefix_name):
            continue
        print("Reading", f)
        with open(f, 'r') as reader:
            lines = [l.strip() for l in reader]
        word_indices = [-1] + [i for i in range(len(lines)) if lines[i].startswith('WORD')]
        find_ending_methods(possible_ending_words, lines)

        print("Trace length:", len(word_indices))
        if len(word_indices) > max_trace_depth():
            val_trace = [e.split()[-1] for e in lines if e.startswith('WORD')]
            while val_trace[0] == starting_char():
                val_trace = val_trace[1:]
            if val_trace[-1] != ending_char():
                val_trace += [ending_char()]
            validation_traces += [tuple(val_trace)]
            print("The trace is too long! Appended to validation data.")
            continue
        #######
        one_trace = []
        for i in range(1, len(word_indices)):
            part = lines[word_indices[i - 1] + 1:word_indices[i] + 1]
            one_trace += [(tuple(part[:-1]), part[-1])]
            method_set |= set([word_str.split()[-1] for (_, word_str) in one_trace])
        # print one_trace
        trace_set.add(tuple(one_trace))
    print("Parsed trace set:", len(trace_set))
    ###############################################################################
    # create legal pairs
    method_list = sorted(list(method_set))
    method2ID = {e: k for (k, e) in enumerate(method_list)}
    actual_next_methods = {w: [0.0 for _ in method_list] for w in method_list}
    for one_trace in trace_set:
        for i in range(1, len(one_trace)):
            previous_word = one_trace[i - 1][-1].split()[-1]
            current_word = one_trace[i][-1].split()[-1]
            actual_next_methods[previous_word][method2ID[current_word]] = 1.0
    ###############################################################################
    instances = set()

    returned_traces = set()
    for one_trace in trace_set:
        feature_trace = []
        visited_method = {w: math.log10(1e-3) for w in method_list}
        for (probs, word_string) in one_trace:
            the_word = word_string.split()[-1]
            next_word_vector = [1.0 if x == the_word else 0.0 for x in method_list]

            feature_trace += [
                (
                    tuple(map(lambda x: visited_method[x], method_list)) +
                    parse_probs_lines(probs, method_list)
                    #    + tuple(next_word_vector)
                    , the_word
                )
            ]
            visited_method[the_word] = math.log10(1.0 - 1e-3)
        one_trace = tuple(feature_trace)
        ####################################################################################
        returned_traces.add(one_trace)
        instances |= set([ps for (ps, _) in one_trace])
    instances = sorted(list(instances))
    instances_dict = {x: str(i) for (i, x) in enumerate(instances)}
    indexed_traces = set()
    for one_trace in returned_traces:
        indexed_traces.add(tuple(
            #map(lambda (features, word): (instances_dict[features], word), one_trace)
            map(lambda x: (instances_dict[x[0]], x[1]), one_trace)
            ))

    return instances, sorted(list(indexed_traces), key=lambda x: len(x)), set(
        validation_traces), method_list, possible_ending_words


def read_ktails_clusters(X, X_id_mapping=None):
    eleID2cluster = {}
    ###
    labels = list(range(len(X)))
    ###
    clusters = {}
    if len(X) != len(labels):
        print("inconsistent number of instances and labels")
        sys.exit(1)
    print("Reading clusters, ID mapping:", len(labels), len(X_id_mapping) if X_id_mapping is not None else 0)
    for the_index in range(len(labels)):
        actual_ID = str(the_index) if X_id_mapping is None else X_id_mapping[str(the_index)]

        cluster_name = 'C' + str(labels[the_index])
        eleID2cluster[actual_ID] = cluster_name
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name] += [actual_ID]
    ##############################################################
    local_mapping = None
    if X_id_mapping is not None:
        local_mapping = {v: k for (k, v) in X_id_mapping.items()}
    ##############################################################
    centroids = {}

    for cls in clusters:
        centroid = [0.0 for _ in range(len(X[0]))]
        for e in clusters[cls]:
            actual_e = e if local_mapping is None else local_mapping[e]
            if len(centroid) != len(X[int(actual_e)]):
                print("ERROR: inconsistent dimension")
                sys.exit(0)
            for i in range(len(centroid)):
                centroid[i] += float(X[int(actual_e)][int(i)])
        for i in range(len(centroid)):
            centroid[i] = float(centroid[i]) / float(len(clusters[cls]))
    return eleID2cluster, centroids, labels


def read_clusters(estimator, X, X_id_mapping=None):
    eleID2cluster = {}
    labels = estimator.labels_
    clusters = {}
    if len(X) != len(labels):
        print("inconsistent number of instances and labels")
        sys.exit(1)
    print("Reading clusters, ID mapping:", len(labels), len(X_id_mapping) if X_id_mapping is not None else 0)
    for the_index in range(len(labels)):
        actual_ID = str(the_index) if X_id_mapping is None else X_id_mapping[str(the_index)]

        cluster_name = 'C' + str(labels[the_index])
        eleID2cluster[actual_ID] = cluster_name
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name] += [actual_ID]
    ##############################################################
    local_mapping = None
    if X_id_mapping is not None:
        local_mapping = {v: k for (k, v) in X_id_mapping.items()}
    ##############################################################
    centroids = {}
    if hasattr(estimator, 'cluster_centers_'):
        for i in range(len(estimator.cluster_centers_)):
            # writer.write('center_' + str(i) + '\t' + '\t'.join([str(x) for x in estimator.cluster_centers_[i]]) + '\n')
            centroids['C' + str(i)] = estimator.cluster_centers_[i]
    else:

        for cls in clusters:
            centroid = [0.0 for _ in range(len(X[0]))]
            for e in clusters[cls]:
                actual_e = e if local_mapping is None else local_mapping[e]
                if len(centroid) != len(X[int(actual_e)]):
                    print("ERROR: inconsistent dimension")
                    sys.exit(0)
                for i in range(len(centroid)):
                    centroid[i] += float(X[int(actual_e)][int(i)])
            for i in range(len(centroid)):
                centroid[i] = float(centroid[i]) / float(len(clusters[cls]))
    return eleID2cluster, centroids, labels


def write_cluster(element2cluster, X, f, X_id_mapping=None):
    if f.endswith('.gz'):
        writer = gzip.open(f, 'wb')
    else:
        writer = open(f, 'w')
    if X_id_mapping is not None:
        local_mapping = {v: k for (k, v) in X_id_mapping.items()}
    else:
        local_mapping = None

    for (id, cluster) in sorted(element2cluster.items(), key=lambda x: x[-1]):
        local_id = id if local_mapping is None else local_mapping[id]
        if int(local_id) >= len(X):
            print("ERROR: out of index")
            print(len(X))
            print(id)
            print(local_mapping)
            print(local_id)
            sys.exit(0)
        writer.write((cluster + '\tID:' + id + '\t' + '\t'.join(map(lambda x: str(x), X[int(local_id)])) + '\n').encode('utf-8'))

    writer.close()


def ending_cluster():
    return 'CEND'


def is_constructor(m):
    return m[0].isalpha() and m[0].isupper() and m[1].isalpha()


def ending_char():
    return '<END>'


def write_trace_cluster_info(id2cluster, generated_traces, output_file):
    for trace_id in range(len(generated_traces)):
        one_trace = generated_traces[trace_id]
        for k in range(len(one_trace)):
            pre_prob_id, pre_word = one_trace[k]
            if k < len(one_trace) - 1:
                post_prob_id, post_word = one_trace[k + 1]
            else:
                if pre_word != lib.ending_char():
                    print("Trace not ending with", lib.ending_char())
                    sys.exit(0)
                post_prob_id, post_word = None, None


def create_fsm(id2cluster, generated_traces):
    startings = set()
    endings = set()
    edges = set()
    endings.add(ending_cluster())
    log = {}
    for one_trace in generated_traces:

        previous_cluster = None
        previous_label = None
        previous_prob_id = None
        for (prob_id, word) in one_trace:
            cluster_name = id2cluster[prob_id]
            if is_constructor(word):
                cluster_name = 'CSTART'
            # if word == ending_char():
            #     cluster_name = ending_cluster()

            if previous_cluster is not None:
                edges.add((previous_cluster, cluster_name, previous_label))
                the_edge = (previous_cluster, cluster_name, previous_label)
                if the_edge not in log:
                    log[the_edge] = set()
                log[the_edge].add((str(previous_prob_id) if previous_prob_id is not None else 'None', str(prob_id)))
            else:
                startings.add(cluster_name)

            previous_cluster = cluster_name
            previous_label = word
            previous_prob_id = prob_id

        ###
        if previous_label == ending_char():
            edges.add((previous_cluster, ending_cluster(), previous_label))
    return StandardAutomata(startings, edges, endings), log


def starting_char():
    return '<START>'


def remove_starting(tr):
    if tr[0] == starting_char():
        tr = tr[1:]
    if tr[-1] != ending_char():
        tr += [ending_char()]
    return tr


def count_accepted_traces(fsm, validation_traces, output_file=None, debug=False):
    """
    Count number of traces in validation_traces accepted by the fsm
    @fsm: input finite state machine
    @validation_traces: input execution traces
    """
    ans = 0
    fsm_adjlst = fsm.create_adjacent_list()
    rejected_traces = []
    for tr in validation_traces:
        flag, rejected_prefices = fsm.is_accepting_one_trace(tr, fsm_adjlst, waiting_time=5)
        if flag:
            ans += 1
        else:
            rejected_traces += [(tuple(tr), rejected_prefices)]  # TODO rejected data could be caused by program crashes

    if output_file is not None and len(rejected_traces) > 0:
        output_file = os.path.abspath(output_file)
        dirname = os.path.dirname(output_file)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if not os.path.isdir(dirname+'/rejected_traces'):
            os.makedirs(dirname+'/rejected_traces')
        with  open(dirname+'/rejected_traces/input.txt','w') as writer:
            writer.write('\n'.join( list( map(lambda x:'<START> '+ ' '.join(x[0]),rejected_traces)) ) +'\n')

        with open(output_file, 'w') as writer:
            for tr, rejected_prefices in rejected_traces:
                writer.write(' '.join(tr) + '\n')
                if rejected_prefices is None:
                    continue
                for tr, node in rejected_prefices:
                    writer.write('\t' + node + '\t' + ' '.join(tr) + '\n')
                writer.write('###################################\n')

    return ans

    # set the timeout handler


def write_X_to_file(X, method_list, traces, f, X_id_mapping=None):
    with open(f, 'w') as writer:
        writer.write('\t'.join(method_list) + '\n')
        id = 0
        for e in X:
            actual_id = str(id) if X_id_mapping is None else X_id_mapping[str(id)]
            writer.write('XID:' + actual_id + '\t' + ' '.join(map(lambda x: str(x), e)) + '\n')
            id += 1
        writer.write('\n')
        id = 0
        for one_trace in traces:
            writer.write('TraceID:' + str(id) + '\t' + '\t'.join(
                #map(lambda (vt_id, word): '(' + str(vt_id) + ',' + word + ')', one_trace)
                map(lambda x: '(' + str(x[0]) + ',' + x[1] + ')', one_trace)
                ) + '\n')
            id += 1


def clustering_pro(args):
    print(args.output_folder)
    # if not os.path.isfile(args.output_folder + '/statistic.txt'):
    #     lib.init_dir(args.output_folder)
    # else:
    #     print "It looks good!"
    #     sys.exit(0)

    lib.init_dir(args.output_folder)

    # collect X
    (X, generated_traces, additional_val_traces, method_list)= parse_sampled_traces(args.generated_traces_folder, 'd')
    print("Training data:", len(generated_traces))

    # read validation trace
    validation_traces = parse_validation_traces(args.validation_traces_folder, 'seed_')
    validation_traces |= additional_val_traces
    print("Validating:", len(validation_traces), "data")

    number_of_clusters = args.num_cluster

    # print args
    if len(X) <= number_of_clusters:
        print("WARNING: number of clusters must be < number of instances!", args.num_cluster, len(X))
        sys.exit(0)

    print("Length of X", len(X))
    # do clustering

    estimator = do_clustering(args, X)
    compute_statistics(X, method_list, args, estimator, generated_traces, validation_traces)


def write_centroids_to_file(centroids, f):
    with open(f, 'w') as writer:
        for (cluster, vt) in centroids.items():
            writer.write(cluster + '\t' + '\t'.join(map(lambda x: str(x), vt)) + '\n')


def write_log_to_file(log_fsm, f):
    with open(f, 'w') as writer:
        for e in log_fsm:
            writer.write(str(e) + '\n')
            writer.write(str(log_fsm[e]) + '\n')
            writer.write('\n')


def create_fsm_for_unit_traces(elementID2cluster, training_traces, output_folder):
    lib.init_dir(output_folder)

    unit_id = 0
    for one_trace in training_traces:
        unit_id += 1
        unit_dir = output_folder + '/fsm_d' + str(unit_id)
        lib.init_dir(unit_dir)
        fsm, log_fsm = create_fsm(elementID2cluster, [one_trace])

        dfa = fsm.nfa2dfa()
        mindfa = graph_lib.minimize_dfa(dfa)

        open(unit_dir + '/fsm.txt', 'w').write(fsm.to_string())
        open(unit_dir + '/dfa.txt', 'w').write(dfa.to_string())
        open(unit_dir + '/mindfa.txt', 'w').write(mindfa.to_string())

        drawing_dot(fsm, unit_dir + '/fsm')
        drawing_dot(dfa, unit_dir + '/dfa')
        drawing_dot(mindfa, unit_dir + '/mindfa')


def extending_ending_states(fsm, ending_methods):
    startings = set(fsm.startings)
    endings = set(fsm.endings)
    edges = set(fsm.transitions)
    for (source, dest, label) in fsm.transitions:
        if label in ending_methods:
            for end in endings:
                edges.add((dest, end, lib.ending_char()))
    return StandardAutomata(startings, edges, endings)


def drawing_dot(fsm, f):
    if len(fsm.states) < 25:
        fsm.to_dot(f, drawing_time=10)


def compute_statistics(X, method_list, args, estimator, generated_traces, validation_traces, output_folder=None,
                       X_id_mapping=None, create_fsm_per_unit_trace=False, ending_methods=None, minimize_dfa=True,
                       ktails=False,check_accepted_traces=True):
    if output_folder is None:
        output_folder = args.output_folder
    lib.init_dir(output_folder)
    if estimator is not None:
        ###
        elementID2cluster, centroids, X_labels = read_clusters(estimator, X, X_id_mapping=X_id_mapping)
    elif ktails:
        # ktails
        elementID2cluster, centroids, X_labels = read_ktails_clusters(X, X_id_mapping=X_id_mapping)
    else:
        print("ERROR: no estimators!")
        sys.exit(0)

    # write cluster info
    write_cluster(elementID2cluster, X, output_folder + '/resultant_cluster.gz', X_id_mapping=X_id_mapping)

    # write centroids
    write_centroids_to_file(centroids, output_folder + '/centroids.txt')

    if create_fsm_per_unit_trace:
        create_fsm_for_unit_traces(elementID2cluster, generated_traces, output_folder + '/unit_fsms')

    # create FSM
    fsm, log_fsm = create_fsm(elementID2cluster, generated_traces)

    # write info of data contained inside each cluster
    write_trace_cluster_info(elementID2cluster, generated_traces, output_folder + '/trace_cluster_info.txt')

    # write fsm log to file
    write_log_to_file(log_fsm, output_folder + '/debug_fsm.txt')

    # DFA
    dfa = fsm.nfa2dfa()
    if check_accepted_traces:
        dfa_num_accepted_traces = count_accepted_traces(dfa, validation_traces,
                                                    output_file=output_folder + '/dfa_uncovered_traces.txt')
    else:
        dfa_num_accepted_traces=-1
    print("Finished validating DFA:", dfa_num_accepted_traces, "validation traces accepted by DFA")

    if minimize_dfa:
        # MinDFA
        mindfa = graph_lib.minimize_dfa(dfa)
    else:
        mindfa = None

    open(output_folder + '/fsm.txt', 'w').write(fsm.to_string())
    drawing_dot(fsm, output_folder + '/fsm')

    open(output_folder + '/dfa.txt', 'w').write(dfa.to_string())
    drawing_dot(dfa, output_folder + '/dfa')


    if minimize_dfa:
        open(output_folder + '/mindfa.txt', 'w').write(mindfa.to_string())

    if minimize_dfa:
        drawing_dot(mindfa, output_folder + '/mindfa')

    # Number of accepted data; size of DFA, MinDFA, FSM;

    # fsm_num_accepted_traces = count_accepted_traces(fsm, validation_traces, debug=True)
    # print "Finished validating FSM:", fsm_num_accepted_traces, "data"

    ###

    # mindfa_num_accepted_traces = count_accepted_traces(mindfa, validation_traces)
    # print "Finished validating MinDFA:", mindfa_num_accepted_traces, "data"

    ##### compute silhouete ####

    # try:
    #     import signal
    #     signal.signal(signal.SIGALRM, lib.handler)
    #     signal.alarm(waiting_time())
    #     silhouette_avg = silhouette_score(np.array(X), estimator.labels_,
    #                                       sample_size=min(
    #                                           args.silhouette_sample_size if args.silhouette_sample_size is not None else len(
    #                                               X), len(X)),
    #                                       random_state=args.seed)
    #     print "silhouette_avg:", silhouette_avg
    #
    #     signal.alarm(0)
    # except TimeoutError:
    #     print "silhouette computation runs too long!"
    #     silhouette_avg = -1
    # except ValueError as e:
    #     print e
    #     silhouette_avg = -1
    # finally:
    #     signal.alarm(0)

    # write statistics
    with open(output_folder + '/statistic.txt', 'w')as writer:
        writer.write('FSM_size:' + '\t' + str(len(fsm.states)) + '\n')
        if dfa is not None:
            writer.write('DFA_size:' + '\t' + str(len(dfa.states)) + '\n')
        if mindfa is not None:
            writer.write('MinDFA_size:' + '\t' + str(len(mindfa.states)) + '\n')
        # writer.write('FSM_validation:' + '\t' + str(fsm_num_accepted_traces) + '\n')
        if dfa_num_accepted_traces is not None:
            writer.write('DFA_validation:' + '\t' + str(dfa_num_accepted_traces) + '\n')
        # writer.write('MinDFA_validation:' + '\t' + str(mindfa_num_accepted_traces) + '\n')
        # writer.write('silhouette_avg:' + '\t' + str(silhouette_avg) + '\n')
        if hasattr(estimator, 'n_clusters'):
            writer.write('num_cluster:\t' + str(estimator.n_clusters) + '\n')
        else:
            n_clusters_ = len(set(X_labels)) - (1 if -1 in X_labels else 0)

            writer.write('num_cluster:\t' + str(n_clusters_) + '\n')
        writer.write('total_validation_traces:\t' + str(len(validation_traces)) + '\n')
        if dfa_num_accepted_traces is not None:
            possible_recall = float(dfa_num_accepted_traces) / float(len(validation_traces))
            writer.write('recall:\t' + str(possible_recall) + '\n')

    ########################
    if ending_methods is not None:
        when_ending_method_available(ending_methods, fsm, output_folder, make_dfa=minimize_dfa)


def when_ending_method_available(ending_methods, fsm, output_folder, make_dfa=False):
    extended_fsm_dir = output_folder + '/extended_endings_fsm'
    lib.init_dir(extended_fsm_dir)

    extended_fsm = extending_ending_states(fsm, ending_methods)
    open(extended_fsm_dir + '/fsm.txt', 'w').write(extended_fsm.to_string())
    drawing_dot(extended_fsm, extended_fsm_dir + '/fsm')

    extended_dfa = extended_fsm.nfa2dfa()
    open(extended_fsm_dir + '/dfa.txt', 'w').write(extended_dfa.to_string())
    drawing_dot(extended_dfa, extended_fsm_dir + '/dfa')

    if make_dfa:
        extended_mindfa = graph_lib.minimize_dfa(extended_dfa)
        open(extended_fsm_dir + '/mindfa.txt', 'w').write(extended_mindfa.to_string())
        drawing_dot(extended_mindfa, extended_fsm_dir + '/mindfa')


def parse_trace_file(f):

    with open(f, 'r') as reader:
        
        lines = [tuple(l.strip().split()) for l in reader if len(l.strip())>0]

        traces = map(lambda tr: tr[1:] if tr[0] == lib.starting_char() else tr, lines)
        return set(traces)

def clustering_step(args,clustering_algorithms = ['kmeans', 'hierarchical']):
    sys.setrecursionlimit(max_trace_depth())

    if os.path.isdir(args.output_folder):
        import shutil
        shutil.rmtree(args.output_folder)

    print(args.output_folder)

    lib.makedirs(args.output_folder)

    # collect X
    X, generated_traces, additional_val_traces, method_list, possible_ending_words = parse_sampled_traces(
        args.generated_traces_folder, 'd')
    print("Training data:", len(generated_traces), "from", args.generated_traces_folder)

    # read validation trace
    validation_traces = set()
    if os.path.isdir(args.validation_traces_folder):
        validation_traces = parse_validation_traces(args.validation_traces_folder, 'seed_')
    elif os.path.isfile(args.validation_traces_folder):
        validation_traces = parse_trace_file(args.validation_traces_folder)
    validation_traces |= additional_val_traces
    print("Validating:", len(validation_traces), "data", "from", args.validation_traces_folder)

    ##########################################################################################

    write_X_to_file(X, method_list, generated_traces, args.output_folder + '/X.txt')
    # print generated_traces
    pool = multiprocessing.Pool(processes=args.args.max_cpu)

    for number_of_clusters in range(args.min_cluster, args.max_cluster):
        for alg in clustering_algorithms:
            if len(X) <= number_of_clusters:
                print("WARNING: number of clusters must be < number of instances!", number_of_clusters, len(X))
                continue

            print("Length of X:", len(X))
            # do clustering
            a=(X, alg, args, generated_traces, method_list, number_of_clusters, validation_traces)
            pool.apply_async(run_cluster,a)
    pool.close()
    pool.join()
    
def run_cluster(X, alg, args, generated_traces, method_list, number_of_clusters, validation_traces):
    output_folder = args.output_folder + '/cls_' + alg + '/S_' + str(number_of_clusters)
    estimator = do_clustering(args, X, ncluster=number_of_clusters, clustering_algorithm=alg,
                                          eps=args.dbscan_eps)
    compute_statistics(X, method_list, args, estimator, generated_traces, validation_traces,
                                   output_folder=output_folder, create_fsm_per_unit_trace=False,
                                   minimize_dfa=args.dfa == 1)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--generated_traces_folder', type=str, default=None,
#                         help='Path to folder of data generated by the trained model')
#     parser.add_argument('--validation_traces_folder', type=str, default='input.txt',
#                         help='Path to file of original training data for validation purpose')
#     parser.add_argument('--alg', type=str, default='kmeans',
#                         help='Clustering algorithm: kmeans, hierarchical clustering')
#     parser.add_argument('--num_cluster', type=int, default='10',
#                         help='input number of clusters')
#     parser.add_argument('--dbscan_eps', type=float, default=0.0001,
#                         help='DBSCAN\'s eps parameter')
#     # parser.add_argument('--seed', type=int, default=9999,
#     #                     help='seed value to initilize Numpy\'s randomState')
#     parser.add_argument('--output_folder', type=str, default=None,
#                         help='Output file to save found clusters, silhouete ')
#     parser.add_argument('--seed', type=int, default=9999,
#                         help='Initialized seed')
#     parser.add_argument('--silhouette_sample_size', type=int, default=None,
#                         help='Silhouette\'s sample size')
#     args = parser.parse_args()

#     sys.setrecursionlimit(max_trace_depth())

#     clustering_pro(args)


def write_ending_methods(ending_methods, outfile):
    with open(outfile, 'w') as writer:
        writer.write('\n'.join(list(ending_methods)))
