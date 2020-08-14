import os
import sys

import lib
from collections import Counter


def select_cluster_traces_by_min_method_coverage(traces, train_method_freqs, method_min_occurrence=2):
    selected_methods_freqs = Counter()
    method_traces_dict = {m: [] for m in train_method_freqs.keys()}
    for index in range(len(traces)):
        for m in traces[index]:
            method_traces_dict[m] += [index]

    selected_traces = []
    is_selected_traces = set()

    for (m, _) in train_method_freqs.most_common()[::-1]:
        for i in range(len(method_traces_dict[m])):
            if selected_methods_freqs[m] >= method_min_occurrence:
                break
            if method_traces_dict[m][i] in is_selected_traces:
                continue

            current_trace = traces[method_traces_dict[m][i]]
            # local_freqs = Counter(current_trace)
            # if local_freqs.most_common()[0][1]>2:
            #     continue
            selected_traces += [current_trace]
            is_selected_traces.add(method_traces_dict[m][i])
            selected_methods_freqs.update(current_trace)

    return selected_methods_freqs, selected_traces


def select_traces_by_coocurrence_pairs(traces):
    co_ocurrence_pairs = set()
    next_to_each_other_pairs = set()

    co_ocurrence_indices = {}
    next_to_each_other_indices = {}

    for tr_id in range(len(traces)):
        tr = traces[tr_id]
        for i in range(len(tr)):
            a = tr[i]
            if i + 1 < len(tr):
                b = tr[i + 1]
                if (lib.is_starting_or_ending_chars(a) and lib.is_starting_or_ending_chars(b)):
                    continue
                update_pair(a, b, next_to_each_other_pairs, next_to_each_other_indices, tr_id)
            ##########################################################################
            for j in range(i + 2, len(tr)):
                b = tr[j]
                if (lib.is_starting_or_ending_chars(a) or lib.is_starting_or_ending_chars(b)):
                    continue
                update_pair(a, b, co_ocurrence_pairs, co_ocurrence_indices, tr_id)
    print("Consecutive pairs of methods:", len(next_to_each_other_pairs))
    print("Co-ocurrence pairs of methods:", len(co_ocurrence_pairs))

    print(co_ocurrence_pairs - next_to_each_other_pairs)

    merge_traces_indices = next_to_each_other_indices

    for p in co_ocurrence_indices:
        if p not in merge_traces_indices:
            merge_traces_indices[p] = co_ocurrence_indices[p]
            print("Adding cooccurence pair:", p)
    interested_pairs = list(merge_traces_indices.keys())

    selected = selecting_traces(interested_pairs, merge_traces_indices, traces)
    output_traces = set([traces[id] for id in selected])
    return output_traces


def selecting_traces(interested_pairs, merge_traces_indices, traces):
    selected = set()
    uncovered_pairs = sorted(list(interested_pairs), key=lambda x: len(merge_traces_indices[x]))
    while len(uncovered_pairs) > 0:
        p = uncovered_pairs[0]
        one_trace = None
        for tr_id in merge_traces_indices[p]:
            if tr_id in selected:
                continue
            one_trace = traces[tr_id]
            break
        if one_trace is None:
            print("Cannot find trace for", p)
            print(selected)
            print(merge_traces_indices[p])
            sys.exit(0)
        print("Handling", p, len(merge_traces_indices[p]), traces[tr_id])
        selected.add(tr_id)
        for i in range(len(one_trace)):
            for j in range(i + 1, len(one_trace)):
                a = one_trace[i]
                b = one_trace[j]
                # if a > b:
                #     a, b = b, a
                if lib.is_starting_or_ending_chars(a) and lib.is_starting_or_ending_chars(b):
                    continue
                try:
                    uncovered_pairs.remove((a, b))
                except ValueError:
                    pass
    return selected


def update_pair(a, b, pair_set, trace_indices, tr_id):
    p = (a, b)

    pair_set.add(p)
    if p not in trace_indices:
        trace_indices[p] = []
    trace_indices[p] += [tr_id]


def simplify_trace(selected_traces):
    ans = set()
    for tr in selected_traces:
        new_tr = []
        for m in tr:
            if len(new_tr) >= 2 and new_tr[-1] == new_tr[-2] == m:
                continue
            new_tr += [m]
        ans.add(tuple(new_tr))
    return ans


def select_traces(input_file, output_file, min_occurrence=5,debug=True):
    with open(input_file, 'r') as reader:
        traces = set([tuple(l.strip().split()) for l in reader])
    method_list=set([e for tr in traces for e in tr])
    ############################################################################
    # data = sorted(list(data), key=lambda x: len(x))
    train_method_freqs = Counter([m for tr in traces for m in tr])
    # median_frequency = int(lib.find_median(map(lambda (m, f): f, train_method_freqs.most_common()))) + 1
    median_frequency = 1+ int(lib.find_median(list(train_method_freqs.values()))) 
    # print(train_method_freqs.most_common()[::-1][-1])

    min_occurrence = min(train_method_freqs.most_common()[::-1][-1][-1], min_occurrence)
    avg_len = [len(tr) for tr in traces]
    avg_len = float(sum(avg_len)) / len(avg_len)
    if debug:
        print("AVG length:", avg_len)
        print("Train method distributions:", train_method_freqs)
        print("Max. possible pairs of methods:", (len(method_list) * (len(method_list) - 1)))
        print("Min. Frequency:", min_occurrence)
    #############################################################################
    # sorting data to determine size of selected data
    # data = sorted(data, key=lambda x: abs(len(x) - avg_len))
    # data = sorted(data, key=lambda x: -len(x))
    traces = sorted(traces, key=lambda x: (len(x), len(set(x)),x))
    # data = sorted(data, key=lambda x: (-len(set(x)),len(x)))
    #############################################################################
    selected_mfreq, selected_traces = select_cluster_traces_by_min_pair_coverage(traces)
    # selected_mfreq, selected_traces = select_cluster_traces_by_min_method_coverage(data,train_method_freqs,method_min_occurrence=2)
    # selected_traces = select_traces_by_coocurrence_pairs(data)
    # selected_traces = simplify_trace(selected_traces)
    ################################################################################
    print("Selected data: ", len(selected_traces))  # , selected_mfreq.most_common()
    with open(output_file, 'w') as writer:
        writer.write('\n'.join(map(lambda x: ' '.join(x), selected_traces)))
    return selected_traces



def select_cluster_traces_by_min_pair_coverage(traces, pair_min_occurrence=1):
    method_pairs_freqs = Counter([(tr[i], tr[i + 1]) for tr in traces for i in range(len(tr) - 1)
                                  # if tr[i] != lib.starting_char() and tr[i + 1] != lib.ending_char()
                                  ])
    print("Covered pairs:", len(method_pairs_freqs))
    pair_traces_dict = {p: [] for p in method_pairs_freqs}
    for k in range(len(traces)):
        tr = traces[k]
        for i in range(len(tr) - 1):
            pair_traces_dict[(tr[i], tr[i + 1])] += [k]

    selected_mfreq = Counter()
    selected_traces = []
    is_selected_traces = set()
    for (p, _) in method_pairs_freqs.most_common()[::-1]:

        for i in range(len(pair_traces_dict[p])):
            if selected_mfreq[p] >= pair_min_occurrence:
                break
            current_trace = traces[pair_traces_dict[p][i]]
            if pair_traces_dict[p][i] not in is_selected_traces:
                selected_traces += [current_trace]
                is_selected_traces.add(pair_traces_dict[p][i])
                selected_mfreq.update(
                    [(current_trace[k], current_trace[k + 1]) for k in range(len(current_trace) - 1)
                     # if current_trace[k] != lib.starting_char() and current_trace[k + 1] != lib.ending_char()
                     ])

    return selected_mfreq, selected_traces


# if __name__ == '__main__':
#     process(DGX1LearningConfiguration())
#     process(UX501LearningConfiguration())
