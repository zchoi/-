from __future__ import print_function

import argparse
import os, random
import utils

import tensorflow as tf
from six.moves import cPickle

from model import Model
import numpy as np


def starting_word():
    return '<START>'


def ending_word():
    return '<END>'


def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return (int(np.searchsorted(t, np.random.rand(1) * s)))


def print_words_probs(words, the_probs):
    arr = [(words[i], x) for (i, x) in enumerate(the_probs)]
    arr = sorted(arr, key=lambda x: x[1], reverse=True)
    return str(arr)


def is_constructor(m):
    return m[0].isalpha() and m[0] == m[0].upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--num_trace', type=int, default=10,
                        help='number of traces to generate')
    parser.add_argument('--prime_text_file', type=str, default=' ',
                        help='prime text file')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed value to initilize Numpy\'s randomState')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output file to save generated text')
    args = parser.parse_args()
    sample(args)


def compute_next_probs(sess, model, words, vocab, state):
    ans = []
    for windex in range(len(words)):
        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(words[windex], 0)
        feed = {model.input_data: x, model.initial_state: state}
        [probs] = sess.run([model.probs], feed)
        p = probs[0]
        ans += [(words[windex], p)]
    return ans


def seed_sample(sess, words, vocab, model, prime_text_file, seed, output_file, min_threshold=0.001, max_length=1000):
    np.random.seed(seed)
    random.seed(seed)

    constructor_indices = [i for i in range(len(words)) if is_constructor(words[i])]

    state = sess.run(model.cell.zero_state(1, tf.float32))
    prime = '<START> ' + open(prime_text_file, 'r').read() + ' <END> <START> '

    for word in prime.split()[:-1]:
        # print(word)
        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word, 0)
        feed = {model.input_data: x, model.initial_state: state}
        [probs, state] = sess.run([model.probs, model.final_state], feed)

    word = prime.split()[-1]
    the_trace = []
    # print('\n')
    # print('\n')
    while True:
        # print('Starting with', word)

        ###

        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word, 0)
        feed = {model.input_data: x, model.initial_state: state}
        [probs, state] = sess.run([model.probs, model.final_state], feed)
        p = probs[0]

        ###
        if word == starting_word():
            #sample = random.choice(constructor_indices)
            sample = weighted_pick(p)
            while not is_constructor(words[sample]):
                sample = weighted_pick(p)
        else:
            #good_methods =[i for i in xrange(len(words)) if not is_constructor(words[i]) and p[i]>=min_threshold and words[i] !=utils.starting_char()]
            #sample = random.choice(good_methods)
            sample = weighted_pick(p)
            while p[sample] < min_threshold or is_constructor(words[sample]):
                sample = weighted_pick(p)

        ### compute pk-tail
        next_ps = compute_next_probs(sess, model, words, vocab, state)

        the_trace += [(p, words[sample], next_ps)]
        word = words[sample]

        ###

        if words[sample] == ending_word():
            break

        if len(the_trace) > max_length:
            break

    with open(output_file, 'w') as writer:
        for (the_p, the_word, next_ps) in the_trace:
            arr = [(words[i], the_p[i]) for i in range(len(the_p))]
            arr = sorted(arr, key=lambda x: x[1], reverse=True)
            writer.write('1-TAIL\t' + '\t'.join([w + ':' + str(p) for (w, p) in arr]) + '\n')

            for (next_word, next_p) in next_ps:
                arr = [(words[i], next_p[i]) for i in range(len(next_p))]
                arr = sorted(arr, key=lambda x: x[1], reverse=True)
                writer.write('2-TAIL\t' + next_word + '\t' + '\t'.join([w + ':' + str(p) for (w, p) in arr]) + '\n')

            writer.write('WORD\t' + the_word + '\n')


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    if args.seed is None:
        # init random state using seed
        np.random.seed(args.seed)
        random.seed(args.seed)

    seed_list = set()
    while len(seed_list) < args.num_trace:
        seed_list.add(random.randint(0, 2 ** 31 - 1))
    utils.init_dir(args.output_folder)
    print(args)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for the_seed in seed_list:
                seed_sample(sess, words, vocab, model, args.prime_text_file, the_seed,
                            args.output_folder + '/seed_' + str(the_seed) + '.txt')


if __name__ == '__main__':
    main()
