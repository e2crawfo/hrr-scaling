"""
For plotting the similarity of vectors in a vector encoding of WordNet,
assuming ID-vectors are not used. Currently has to be run from inside
hrr-scaling/python directory.
"""

from hrr_scaling.tools.bootstrap import Bootstrapper
from hrr_scaling.tools import fh
from hrr_scaling.corpora_management import VectorizedCorpus

import random

import numpy as np
import matplotlib.pyplot as plt


dimension = 512
unitary = False
id_vecs = False
normalize = True
input_dir = './wordnet_data/'
proportion = 1.0
num_synsets = -1
num_samples = 100000


def generate_data(filename=None, unitary=False, id_vecs=False,
                  normalize=True, save=False, sp_noise=0):

    bs = Bootstrapper(write_raw_data=True)

    config_dict = {'S': num_samples, 'N': num_synsets, 'U': unitary,
                   'Norm': normalize, 'id': id_vecs}

    if not filename:
        filename = fh.make_filename(
            'boot_results', use_time=False, config_dict=config_dict)

    try:
        bs.read_bootstrap_file(filename)
    except:
        corpus = VectorizedCorpus(
            dimension, input_dir, unitary_relations=unitary,
            normalize=normalize, proportion=proportion,
            num_synsets=num_synsets, id_vecs=id_vecs, sp_noise=sp_noise)

        vectors = corpus.id_vectors

        vector_keys = vectors.keys()
        size = len(vector_keys)

        for i in xrange(num_samples):
            s1 = s2 = random.randrange(0, size)

            while s1 == s2:
                s2 = random.randrange(0, size)

            key1 = vector_keys[s1]
            key2 = vector_keys[s2]
            similarity = np.dot(vectors[key1], vectors[key2])

            bs.add_data('sim', similarity)

        if save:
            bs.print_summary(filename)

    return bs


def plot_data(bs, plot_filename='simplot.pdf'):

    plt.hist(bs.data['sim'], bins=1000)
    plt.xlabel('Dot product')
    plt.gca().set_xlim((-1.0, 1.0))
    plt.gca().set_ylim((0, 1500))
    plt.savefig(plot_filename)


def run():

    normalize = True


    # ------------------

    unitary = False
    id_vecs = False

    bs = generate_data(
        unitary=unitary, id_vecs=id_vecs, normalize=normalize, save=True)

    data = np.reshape(bs.data['sim'], (1, -1))

    # ------------------

    unitary = True
    id_vecs = False

    bs = generate_data(
        unitary=unitary, id_vecs=id_vecs, normalize=normalize, save=True)

    temp = np.reshape(bs.data['sim'], (1, -1))
    data = np.concatenate((data, temp))

    # ------------------

    unitary = False
    id_vecs = True

    bs = generate_data(
        unitary=unitary, id_vecs=id_vecs, normalize=normalize, save=True)

    temp = np.reshape(bs.data['sim'], (1, -1))
    data = np.concatenate((data, temp))

    nbins = 200
    bins = np.linspace(-1, 1, nbins)
    alpha = 0.3

    fig = plt.figure(figsize=(7,5))

    plt.subplot(3, 1, 1)
    plt.hist(
        data[2], bins=bins, alpha=alpha, color='b')
    plt.gca().set_xlim((-1.1, 1.1))
    plt.gca().set_ylim((0, 10000))
    plt.gca().set_yticks((0, 5000, 10000))
    plt.ylabel('Freq.')
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.gca().text(0.01, .8, r"\textbf{A}", fontsize=15, transform=plt.gca().transAxes)

    plt.subplot(3, 1, 2)
    plt.hist(
        data[0], bins=bins, alpha=alpha, color='r')
    plt.gca().set_xlim((-1.1, 1.1))
    plt.gca().set_ylim((0, 10000))
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.gca().set_yticks((0, 5000, 10000))
    plt.ylabel('Freq.')
    plt.gca().text(0.01, .8, r"\textbf{B}", fontsize=15, transform=plt.gca().transAxes)

    plt.subplot(3, 1, 3)
    plt.hist(
        data[1], bins=bins, alpha=alpha, color='g')
    plt.gca().set_xlim((-1.1, 1.1))
    plt.gca().set_ylim((0, 10000))
    plt.gca().set_yticks((0, 5000, 10000))
    plt.gca().text(0.01, .8, r"\textbf{C}", fontsize=15, transform=plt.gca().transAxes)
    plt.xlabel('Dot product')
    plt.ylabel('Freq.')

    plt.subplots_adjust(
        top=0.94, right=0.89, bottom=0.1, left=0.11, hspace=0.2, wspace=0.2)
    plt.savefig('sim_plot.pdf')

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test an HRR extractor.')

    parser.add_argument(
        '--unitary', dest='unitary', action='store_true',
        help='Supply this argument to make relation-type vectors unitary')

    parser.add_argument(
        '--id-vecs', dest='id_vecs', action='store_true',
        help='Supply this argument to use random ID-vectors')

    parser.add_argument(
        '--normalize', dest='normalize', action='store_true',
        help='Supply this argument to normalize the semantic pointers')

    parser.add_argument(
        '--sp-noise', dest='sp_noise', default=0, type=int,
        help="Change the amount of noise added to the semantic pointers")

    argvals = parser.parse_args()
    unitary = argvals.unitary
    id_vecs = argvals.unitary
    normalize = argvals.normalize
    sp_noise = argvals.sp_noise

    bs = generate_data(
        filename=None, unitary=unitary, id_vecs=id_vecs,
        normalize=normalize, sp_noise=sp_noise)

    plot_data(bs)
