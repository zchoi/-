import sys, os, shutil, itertools, random
import numpy as np

def find_files(d, name):
    if os.path.isdir(d):
        ans = []
        for c in os.listdir(d):
            ans += find_files(d + '/' + c, name)
        return ans
    elif os.path.isfile(d):
        if os.path.basename(d) == name:
            return [d]
        else:
            return []


# def find_folders_by_prefix(d, prefix):
#     if os.path.isdir(d):
#         ans = []
#         if os.path.basename(d).startswith(prefix):
#             ans += [d]
#         for c in os.listdir(d):
#             ans += find_folders_by_prefix(d + '/' + c, prefix)
#         return ans
#     else:
#         return []

def find_folders_by_prefix(path, prefix):
    ans = []
    for root, dirs, files in os.walk(path):
        # for file in files:
        #     # if file.endswith(".csv"):
        #     #     path_file = os.path.join(root, file)
        #     #     print(path_file)
        if os.path.basename(root).startswith(prefix):
            ans += [root]
    return ans


def find_folders_by_postfix(path, postfix):
    ans = []
    for root, dirs, files in os.walk(path):
        # for file in files:
        #     # if file.endswith(".csv"):
        #     #     path_file = os.path.join(root, file)
        #     #     print(path_file)
        if os.path.basename(root).endswith(postfix):
            ans += [root]
    return ans


def find_folders(path, name):
    ans = []
    for root, dirs, files in os.walk(path):
        # for file in files:
        #     # if file.endswith(".csv"):
        #     #     path_file = os.path.join(root, file)
        #     #     print(path_file)
        if os.path.basename(root) == name:
            ans += [root]
    return ans


def makedirs(d):
    if not os.path.isdir(d):
        os.makedirs(d)


def make_parents_dir(f):
    p = os.path.dirname(f)
    makedirs(p)


def find_files_by_prefix(d, prefix):
    if os.path.isdir(d):
        ans = []
        for c in os.listdir(d):
            ans += find_files_by_prefix(d + '/' + c, prefix)
        return ans
    elif os.path.isfile(d):
        if os.path.basename(d).startswith(prefix):
            return [d]
        else:
            return []


def find_median(lst):
    lst = sorted(lst)
    if len(lst) % 2 == 1:
        return lst[len(lst) // 2]
    else:

        return (lst[len(lst) // 2] + lst[len(lst) // 2 - 1]) / 2


def find_files_by_suffix(d, suffix):
    # if os.path.isdir(d):
    #     ans = []
    #     for c in os.listdir(d):
    #         ans += find_files_by_suffix(d + '/' + c, suffix)
    #     return ans
    # elif os.path.isfile(d):
    #     if os.path.basename(d).endswith(suffix):
    #         return [d]
    #     else:
    #         return []

    ans = []
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith(suffix):
                path_file = os.path.join(root, file)
                ans += [path_file]

    return ans


def init_dir(d):
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)


def avg_of(d):
    if len(d) == 0:
        return 0.0
    else:
        return float(sum(d)) / float(len(d))


def ending_char():
    return '<END>'


def starting_char():
    return '<START>'


def is_starting_or_ending_chars(c):
    return c == starting_char() or c == ending_char()


def read_input_trace_file(f):
    lines = [l.strip() for l in open(f, 'r')]
    traces = [tuple(y) for k, y in itertools.groupby(lines, lambda z: z == starting_char() or z == ending_char()) if
              not k]
    return traces


def remove_1_strings_patterns(tr, max_repeated_length=4):
    ans = []
    for x in tr:
        recent_ones = ans[-max_repeated_length:]
        if len(recent_ones) >= max_repeated_length and len(set(recent_ones)) == 1:
            continue
        ans += [x]
    return tuple(ans)


# def remove_2_strings_patterns(tr, max_repeated_length=4):
#     selected = [True for _ in xrange(len(tr))]
#     for i in xrange(1,len(tr)):
#
#         two_strings = tr[i-1:i+1]
#         k=i-2
#         cnt=0
#         while k>=0:
#             st = tr[k-1:k+1]
#             if st!=two_strings:
#                 break
#             else:
#                 cnt+=1
#             k-=2
#     return tuple(ans)

def weighted_pick(weights):
    import numpy as np
    t = np.cumsum(weights)
    s = np.sum(weights)
    return (int(np.searchsorted(t, np.random.rand(1) * s)))


def randomly_pick(probs):
    sumarr = [0.0]
    s = 0.0

    for e in probs:
        s += e
        sumarr += [s]
    sumarr = [float(e) / float(s) for e in sumarr]
    x = random.random()

    for i in range(1, len(sumarr)):
        low = sumarr[i - 1]
        up = sumarr[i]
        if low <= x <= up:
            return i - 1
    # print "ERROR:"
    # print probs
    # print sumarr
    # print x
    return None


def starts_with_prefices(cluster_file, cluster_prefices):
    for p in cluster_prefices:
        if cluster_file.startswith(p):
            return True
    return False


def remove_extension(name):
    arr = name.split('.')
    return '.'.join(arr[:-1])


class TimeoutError(Exception):
    pass


def handler(signum, frame):
    print('Signal handler called with signal', signum)
    raise TimeoutError('The function was running for too long! Terminated!')


def is_data_extendable_project(api):
    the_list = ["duy_java_util_DuyArrayList", "duy_java_util_DuyHashMap", "duy_java_util_DuyHashSet",
                "duy_java_util_DuyHashtable", "duy_java_util_DuyLinkedList", "duy_java_util_DuyStringTokenizer",
                "java_net_Socket", "java_security_Signature", "java_util_zip_ZipOutputStream",
                "net_sf_jftp_net_wrappers_SftpConnection",
                "org_apache_xalan_templates_ElemNumber_NumberFormatStringTokenizer",
                "org_apache_xml_serializer_ToHTMLStream", "org_columba_ristretto_smtp_SMTPProtocol"]
    for x in the_list:
        if api in x:
            return True
    return False


def max_label_repeated_per_trace():
    return 2


def overall_min_label_coverage():
    return 10


def max_eval_seeds():
    return 10
