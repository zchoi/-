from __future__ import print_function

import argparse
import os, sys,time,random

import tensorflow as tf
from six.moves import cPickle

from model import Model
import numpy as np
import trace_sample
import utils
from collections import Counter
import multiprocessing

def generate_probs(sess, model, words, vocab, tr, output_file):
    print(output_file)

    for e in tr:
        if ' ' in e:
            print("WARNING: input trace might not be in good format!")
            sys.exit(0)
    if tr[0] != utils.starting_char():
        tr = [utils.starting_char()] + tr

    if tr[-1] != utils.ending_char():
        tr += [utils.ending_char()]
    #############
    #max_pattern_to_check = 1 + len(tr) / 2
    #for k in xrange(max_pattern_to_check, 0, -1):
    #    tr = shortening_sequence(tr, pattern_length=k)
    #tr = limit_trace(tr, max_repetition=3)
    #############
    state = sess.run(model.cell.zero_state(1, tf.float32))
    prime = tr + [utils.starting_char()]

    for word in prime:
        # print(word)
        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word, 0)
        feed = {model.input_data: x, model.initial_state: state}
        [probs, state] = sess.run([model.probs, model.final_state], feed)

    #############

    previous_probs = probs
    entries = []
    for the_index in range(1, len(tr)):
        word = tr[the_index]

        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word, 0)
        feed = {model.input_data: x, model.initial_state: state}
        [probs, state] = sess.run([model.probs, model.final_state], feed)

        ######

        next_two_tails = trace_sample.compute_next_probs(sess, model, words, vocab, state)
        entries += [(previous_probs[0], word, next_two_tails)]

        ######
        previous_probs = probs

    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, 'w') as writer:
        for (the_p, the_word, next_ps) in entries:
            arr = [(words[i], the_p[i]) for i in range(len(the_p))]
            arr = sorted(arr, key=lambda x: x[1], reverse=True)
            writer.write('1-TAIL\t' + '\t'.join([w + ':' + str(p) for (w, p) in arr]) + '\n')

            for (next_word, next_p) in next_ps:
                arr = [(words[i], next_p[i]) for i in range(len(next_p))]
                arr = sorted(arr, key=lambda x: x[1], reverse=True)
                writer.write('2-TAIL\t' + next_word + '\t' + '\t'.join([w + ':' + str(p) for (w, p) in arr]) + '\n')

            writer.write('WORD\t' + the_word + '\n')


def sample(args,input_traces,output_feature_file):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    tf.reset_default_graph()

    model = Model(saved_args, True)


    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            generate_probs(sess, model, words, vocab, input_traces, output_feature_file)

def feature_engineering(input_option):
    print("Feature engineering")
    if os.path.isdir(input_option.features4clustering_dir):
        import shutil
        shutil.rmtree(input_option.features4clustering_dir)
    os.makedirs(input_option.features4clustering_dir)
    with  open(input_option.cluster_trace_file,'r') as reader:    
        traces=[l.strip().split() for l in reader]
    index=0
    pool = multiprocessing.Pool(processes=input_option.args.max_cpu)
    for tr in traces:
        index+=1
        a=(input_option.args,tr,input_option.features4clustering_dir+'/d'+str(index)+'.txt')
        pool.apply_async(sample, a)
    pool.close()
    pool.join()
