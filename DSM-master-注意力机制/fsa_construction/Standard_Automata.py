import sys, os, random

from graphviz import Digraph
from collections import deque, Counter
import signal
from lib import TimeoutError
import lib


def phi_state():
    return 'PHI'


def bfs(n, adjlist):
    q = deque()
    q.append(n)
    visited = set()
    visited.add(n)
    while len(q) > 0:
        n = q.popleft()
        if n not in adjlist:
            continue
        for v in adjlist[n]:
            if v not in visited:
                visited.add(v)
                q.append(v)
    return visited
    return element2cluster


def is_accepted(source, index, tr, adjlst, debug=False):
    if index == len(tr):
        return True

    current_label = tr[index]
    if debug:
        print("\t", source, index, tr[index], tr, adjlst[source])
    for (dest, label) in adjlst[source]:
        if label != current_label:
            continue
        if debug:
            print("\t\tFound", (dest, label))
        if is_accepted(dest, index + 1, tr, adjlst, debug=debug):
            return True

    return False


def is_accepted_bfs(init_node, tr, adjlst, debug=False):
    current_nodes = set([init_node])

    for current_index in range(len(tr)):
        current_label = tr[current_index]
        next_nodes = set()
        for source in current_nodes:
            for (dest, label) in adjlst[source]:
                if label != current_label:
                    continue
                if debug:
                    print("\t\tFound", (dest, label))
                next_nodes.add(dest)
        if len(next_nodes) == 0 and current_index < len(tr) - 1:
            return False, (tr[:current_index + 1],init_node)
        current_nodes = next_nodes
    return True, None


def create_clusters(pairs, all_states):
    adjlist = {n: set() for n in all_states}
    for (u, v) in pairs:
        adjlist[u].add(v)
        adjlist[v].add(u)
    element2cluster = {}
    for n in all_states:
        visited = bfs(n, adjlist)
        cluster_name = sorted(list(visited))[0]
        for e in visited:
            element2cluster[e] = 'C' + cluster_name
    clusters = {}
    for e in element2cluster:
        if element2cluster[e] not in clusters:
            clusters[element2cluster[e]] = []
        clusters[element2cluster[e]] += [e]
    # print clusters
    return element2cluster


class StandardAutomata:
    def __init__(self, startings, edges, endings):
        if len(endings) == 0:
            print("WARNING: no ending states")

        self.transitions = set(edges)
        self.startings = set(startings)
        self.endings = set(endings)
        self.states = set()
        for p in edges:
            # if len(p)!=3:
            #     print("ERROR:",p)
            (source, dest, label)=p
            self.states.add(source)
            self.states.add(dest)

    def clone(self):
        return StandardAutomata(self.startings,self.transitions,self.endings)

    def remove_unconnected_states_to_endings(self):
        rev_adjlist = {}
        all_nodes = set()
        for e in self.transitions:
            source = e[0]
            dest = e[1]
            all_nodes.add(source)
            all_nodes.add(dest)
            if dest not in rev_adjlist:
                rev_adjlist[dest] = []
            rev_adjlist[dest] += [source]
        # bfs
        visted = set()
        for x in self.endings:
            if x not in visted:
                found = bfs(x, rev_adjlist)
                visted.update(found)

        # list unreachable states

        unreached_nodes = all_nodes - visted

        return self.remove_state(unreached_nodes)

    def remove_unreachable_states_from_starting(self):
        # bfs to find reachable states
        adjlist = {}
        all_nodes = set()
        for e in self.transitions:
            source = e[0]
            dest = e[1]
            all_nodes.add(source)
            all_nodes.add(dest)
            if source not in adjlist:
                adjlist[source] = []
            adjlist[source] += [dest]
        # bfs
        visted = set()
        for x in self.startings:
            if x not in visted:
                found = bfs(x, adjlist)
                visted.update(found)

        # list unreachable states

        unreached_nodes = all_nodes - visted
        return self.remove_state(unreached_nodes)

    def to_string(self):
        outlines = [len(self.startings)]
        for n in self.startings:
            outlines += [n]
        outlines += [len(self.endings)]
        for n in self.endings:
            outlines += [n]
        outlines += [str(len(self.transitions))]
        for e in self.transitions:
            outlines += [e[0] + '\t' + e[1] + '\t' + e[2]]
        return '\n'.join([str(x) for x in outlines])

    def find_delta(self, node):
        f = {}
        for (source, dest, label) in self.transitions:

            for v in node:
                if source != v:
                    continue
                if label not in f:
                    f[label] = set()
                f[label].add(dest)
        ans = {}
        for e in f.items():
            ans[e[0]] = tuple(sorted(list(e[1])))
        return ans

    def nfa2dfa(self):
        print("Input states:", len(self.states))
        #########################################################################################
        q = deque()
        found_states = set()
        delta_function = {}
        for s in self.startings:
            q.append((s,))
            found_states.add((s,))
        while len(q) > 0:
            n = q.popleft()
            delta = self.find_delta(n)
            delta_function[n] = delta
            # print n, delta
            for x in delta.values():
                if x not in found_states:
                    q.append(x)
                    found_states.add(x)
        print("Found state:", len(found_states))
        ##########################################################################################
        starting_states = set([n for n in found_states for e in self.startings if e in n])
        ending_states = set([n for n in found_states for e in self.endings if e in n])
        dfa_transitions = set()
        for n in found_states:
            source = n
            for e in delta_function[n].items():
                label = e[0]
                dest = e[1]
                dfa_transitions.add((source, dest, label))
        ###########################################################################################

        state_index = sorted(list(found_states))
        state_mapping = {state_index[i]: str(i) for i in range(len(found_states))}

        g = StandardAutomata(list(map(lambda x: state_mapping[x], starting_states)),
                             list(map(lambda x: (state_mapping[x[0]], state_mapping[x[1]], x[2]), dfa_transitions)),
                             list(map(lambda x: state_mapping[x], ending_states)))
        return g

    def extend_dfa(self, to_extend_state=phi_state()):
        startings = set(self.startings)
        endings = set(self.endings)
        label_set = set([label for (_, _, label) in self.transitions])
        adjlst = {n: set() for n in self.states}
        transitions = set(self.transitions)
        for (source, dest, label) in self.transitions:
            if source not in adjlst:
                adjlst[source] = set()
            adjlst[source].add(label)
        for n in adjlst:
            for l in label_set:
                if l not in adjlst[n]:
                    transitions.add((n, to_extend_state, l))
        for l in label_set:
            transitions.add((to_extend_state, to_extend_state, l))
        return StandardAutomata(startings, transitions, endings)

    def minimize_dfa(self):
        print("Input states:", len(self.states))
        label_set = set([e[2] for e in self.transitions])
        ################################################################################
        rev_delta = {n: {l: set() for l in label_set} for n in self.states}
        self_adjlist = {n: set() for n in self.states}
        # computing delta
        for (source, dest, label) in self.transitions:
            self_adjlist[source].add((dest, label))
        for source in self_adjlist:
            for t in self_adjlist[source]:
                dest = t[0]
                l = t[1]
                rev_delta[dest][l].add(source)
        ################################################################################
        marked = set([(u, v) for u in self.states for v in self.states if u < v and (
                (u in self.endings and v not in self.endings) or (
                u not in self.endings and v in self.endings))])
        the_queue = deque()
        for p in marked:
            the_queue.append(p)
        print("Searching ...")

        while len(the_queue) > 0:
            n = the_queue.popleft()
            a = n[0]
            b = n[1]
            # print n
            for l in label_set:
                for u in rev_delta[a][l]:
                    for v in rev_delta[b][l]:
                        if u < v and (u, v) not in marked:
                            marked.add((u, v))
                            the_queue.append((u, v))
                        if v < u and (v, u) not in marked:
                            marked.add((v, u))
                            the_queue.append((v, u))

        ################################################################################
        print("Marked:", len(marked))
        unmarked = [(u, v) for u in self.states for v in self.states if u < v and (u, v) not in marked]
        print("Unmarked:", len(unmarked))

        element2cluster = create_clusters(unmarked, self.states)
        # print set(element2cluster.values())
        min_transitions = set()
        for (source, dest, label) in self.transitions:
            min_transitions.add((element2cluster[source], element2cluster[dest], label))

        min_startings = list(map(lambda x: element2cluster[x], self.startings))
        min_endings = list(map(lambda x: element2cluster[x], self.endings))

        g = StandardAutomata(min_startings, min_transitions, min_endings)
        print("Min. state:", len(g.states))
        return g

    def remove_state(self, to_remove_states):
        print("Removing the following states:", to_remove_states)

        startings = set(self.startings)
        endings = set(self.endings)
        edges = set()
        for x in to_remove_states:
            if x in startings:
                startings.remove(x)
            if x in endings:
                endings.remove(x)

        for e in self.transitions:
            source = e[0]
            dest = e[1]
            if source in to_remove_states or dest in to_remove_states:
                continue
            edges.add(e)
        return StandardAutomata(startings, edges, endings)

    def to_dot(self, filename, drawing_time=None):

        f = Digraph('Automata', format='eps')
        f.body.extend(['rankdir=LR', 'size="8,5"'])
        f.attr('node', shape='star')
        for n in self.startings:
            f.node(n)
        f.attr('node', shape='doublecircle')
        for n in self.endings:
            f.node(n)
        f.attr('node', shape='circle')
        for (source, dest, label) in self.transitions:
            f.edge(source, dest, label=label)

        if filename is not None:
            try:
                if drawing_time is not None:
                    signal.signal(signal.SIGALRM, lib.handler)
                    signal.alarm(drawing_time)
                f.render(filename, view=False)
            except TimeoutError as e:
                print("Drawing dot:", e)
            finally:
                if drawing_time is not None:
                    signal.alarm(0)
        return f.source

    # def to_string(self):
    #     outlines = [len(self.startings)]
    #     for n in self.startings:
    #         outlines += [n]
    #     outlines += [len(self.endings)]
    #     for n in self.endings:
    #         outlines += [n]
    #     outlines += [str(len(self.transitions))]
    #     for e in self.transitions:
    #         outlines += [e[0] + '\t' + e[1] + '\t' + e[2]]
    #     return '\n'.join([str(x) for x in outlines])

    def is_accepting_one_trace(self, tr, adjlst, debug=False, waiting_time=10):
        try:
            if waiting_time is not None:
                signal.signal(signal.SIGALRM, lib.handler)
                signal.alarm(waiting_time)
            rejected_prefices = []
            for s in self.startings:
                # if is_accepted(s, 0, tr, adjlst, debug=debug):
                #     return True
                flag, rejected_prefix = is_accepted_bfs(s, tr, adjlst, debug=debug)
                if flag:
                    return flag, None
                else:
                    rejected_prefices += [rejected_prefix]

            return False, rejected_prefices
        except Exception as e:
            print(e)
            return False,None
        finally:
            signal.alarm(0)

    def find_random_walk(self, source, cur_trace, local_label_coverage, max_length, max_label_coverage_per_trace,
                         adjlist, visited_states):
        if len(cur_trace) > max_length:
            return None
        visited_states.add(source)
        if source in self.endings:
            p_to_exit = 1.0 / (1.0 + len(adjlist[source]))
            if random.random() <= p_to_exit:
                return tuple(cur_trace)
        next_ones = list(adjlist[source])
        random.shuffle(next_ones)

        while len(next_ones) > 0:
            index = random.randint(0, len(next_ones) - 1)
            (next_dest, next_method) = next_ones[index]

            if next_dest not in visited_states and (max_label_coverage_per_trace is None or local_label_coverage[
                next_method] < max_label_coverage_per_trace):
                et = (source, next_method, next_dest)

                visited_states.add(next_dest)

                cur_trace += [et]
                local_label_coverage[next_method] += 1

                random_path = self.find_random_walk(next_dest, cur_trace, local_label_coverage, max_length,
                                                    max_label_coverage_per_trace, adjlist, visited_states)

                if random_path is not None:
                    return random_path

                cur_trace.pop()
                local_label_coverage[next_method] -= 1

            del next_ones[index]

        return None

    def create_adjacent_list(self):
        ans = {n: [] for n in self.states}
        for (source, dest, label) in self.transitions:
            ans[source] += [(dest, label)]
        for n in self.states:
            ans[n] = sorted(list(set(ans[n])))
        return ans

    def randomly_generate_one_trace(self, adjlist, max_label_coverage_per_trace, max_length=1000):

        starting_states = list(self.startings)
        random.shuffle(starting_states)
        for starting in starting_states:
            the_trace = []
            local_label_coverage = Counter()
            visited_states = set()
            t = self.find_random_walk(starting, the_trace, local_label_coverage, max_length,
                                      max_label_coverage_per_trace, adjlist, visited_states)

            if t is not None:
                return t

                # def to_FAdo_nfa(self):
                #     fsm = NFA()
                #     self.init_FAdo(fsm)
                #     return fsm
                #
                # def init_FAdo(self, fsm):
                #     fsm.setSigma(list(set([l for (_, _, l) in self.transitions])))
                #     all_states = set([source for (source, _, _) in self.transitions] + [dest for (_, dest, _) in self.transitions])
                #     all_states = sorted(list(all_states))
                #     state_indices = {x: i for (i, x) in enumerate(all_states)}
                #     for state in all_states:
                #         fsm.addState(state)
                #     fsm.setInitial([state_indices[s] for s in self.startings])
                #     for s in self.endings:
                #         fsm.addFinal(state_indices[s])
                #     for (source, dest, label) in self.transitions:
                #         fsm.addTransition(state_indices[source], label, state_indices[dest])
                #     return fsm
                #
                # def to_FAdo_dfa(self):
                #     fsm = DFA()
                #     self.init_FAdo(fsm)
                #     return fsm


def minimize_dfa(dfa):
    extended_dfa = dfa.extend_dfa()
    extended_dfa = extended_dfa.remove_unreachable_states_from_starting()
    mindfa = extended_dfa.minimize_dfa()
    # mindfa = mindfa.remove_unconnected_states_to_endings()
    mindfa = mindfa.remove_state(set(['C' + phi_state()]))
    return mindfa


#
# def render_dot_source(st, f):
#     src = Source(st, format='eps')
#     src.render(f, view=False)
#
#
# def FAdo_minimize_dfa(nfa,debug=False):
#     FAdo_nfa = nfa.to_FAdo_nfa()
#
#     FAdo_dfa = FAdo_nfa.toDFA()
#
#     if deque==False and not FAdo_dfa.completeP():
#         print "Before: FAdo_dfa is not complete"
#
#     FAdo_dfa = FAdo_dfa.complete()
#
#     if debug and not FAdo_dfa.completeP():
#         print "After: FAdo_dfa is not complete"
#
#     FAdo_mindfa = FAdo_dfa.minimal(complete=False)
#
#     if debug and not FAdo_dfa.equivalentP(FAdo_mindfa):
#         print "DFA and minDFA are not equivelent"
#     if debug:
#         print "Finish!"
#         render_dot_source(FAdo_nfa.dotFormat(), 'nfa')
#         render_dot_source(FAdo_dfa.dotFormat(), 'dfa')
#         render_dot_source(FAdo_mindfa.dotFormat(), 'mindfa')


def parse_fsm_file(model_file, read_stops_file='stops.txt'):
    lines = [l.strip() for l in open(model_file, 'r')]
    cnt = 0
    #######################################
    n_startings = int(lines[cnt])
    cnt += 1
    startings = set()
    for _ in range(n_startings):
        startings.add(lines[cnt])
        cnt += 1
    #######################################
    n_endings = int(lines[cnt])
    cnt += 1
    endings = set()
    for _ in range(n_endings):
        endings.add(lines[cnt])
        cnt += 1
    #######################################
    try:
        n_transitions = int(lines[cnt])
        cnt += 1
    except ValueError:
        n_transitions = len(lines) - cnt
    edges = []
    for _ in range(n_transitions):
        e = lines[cnt].split()[:3]

        edges += [tuple(e)]
        cnt += 1
    if read_stops_file is not None:
        try:
            endings |= set(
                [l.strip() for l in open(os.path.dirname(os.path.abspath(model_file)) + '/' + read_stops_file, 'r')])
        except IOError as e:
            print( os.path.dirname(os.path.abspath(model_file)) + '/' + read_stops_file, "is not available")
    return StandardAutomata(startings, edges, endings)
