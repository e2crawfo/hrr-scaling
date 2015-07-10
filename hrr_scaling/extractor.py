from hrr_scaling.tools.hrr import HRR

import heapq
import random
import collections
import numpy as np


class Extractor(object):

    # the associative memory maps from index_vectors to stored_vectors
    # index_vectors and stored_vectors must both be OrderedDicts whose values
    # are vectors
    def __init__(self, index_vectors, stored_vectors, bootstrapper,
                 threshold=0.3, output_dir='.'):

        self.index_vectors = index_vectors
        self.stored_vectors = stored_vectors
        self.threshold = threshold
        self.dimension = len(index_vectors.values()[0])
        self.num_items = len(index_vectors)
        self.hrr_vecs = collections.OrderedDict(
            [(key, HRR(data=self.index_vectors[key]))
             for key in self.index_vectors])

        self.similarities = collections.OrderedDict(
            zip(self.index_vectors, [0 for i in range(len(index_vectors))]))

        self.return_vec = True

        self.bootstrapper = bootstrapper
        self.output_dir = output_dir

    def add_data(self, *args):
        self.bootstrapper.add_data(*args)

    def extract(self, item, query, target_keys=None):

        if target_keys:
            self.print_instance_difficulty(item, query)

        item_hrr = HRR(data=item)
        query_hrr = HRR(data=query)
        noisy_hrr = item_hrr.convolve(~query_hrr)
        return self.associate(noisy_hrr.v, target_keys)

    def associate(self, noisy_vector, target_keys=None):

        keys = self.index_vectors.keys()

        for key in keys:
            self.similarities[key] = np.dot(
                noisy_vector, self.index_vectors[key])

        result = np.zeros(self.dimension)

        for key in keys:
            sim = self.similarities[key]
            if sim > self.threshold:
                result += self.stored_vectors[key]

        results = [result]

        # Bookkeeping
        if target_keys:
            num_correct_relations = len(target_keys)

            for key in target_keys:
                self.add_data(
                    "correct_dot_product", self.similarities[key])

            nlargest = heapq.nlargest(
                num_correct_relations + 1,
                self.similarities.iteritems(),
                key=lambda x: x[1])

            largest_incorrect = filter(
                lambda x: x[0] not in target_keys, nlargest)[0]

            self.add_data(
                "largest_incorrect_dot_product", largest_incorrect[1])

            reached_threshold = filter(
                lambda x: self.similarities[key] > self.threshold, keys)

            self.add_data("n_reaching_threshold", len(reached_threshold))

        return results

    def print_instance_difficulty(self, item, query, target_keys):

        if target_keys:
            # Print data about how difficult the current instance is
            correct_key = target_keys[0]

            item_hrr = HRR(data=item)
            query_hrr = HRR(data=query)
            noisy_hrr = item_hrr.convolve(~query_hrr)

            correct_hrr = HRR(data=self.index_vectors[correct_key])
            sim = noisy_hrr.compare(correct_hrr)
            dot = np.dot(noisy_hrr.v, correct_hrr.v)
            norm = np.linalg.norm(noisy_hrr.v)
            print "Ideal similarity: ", sim
            print "Ideal dot: ", dot
            print "Ideal norm: ", norm

            self.ideal_dot = dot

            hrrs = [(key, HRR(data=iv))
                    for key, iv in self.index_vectors.iteritems()
                    if key != correct_key]

            sims = [noisy_hrr.compare(h) for (k, h) in hrrs]
            dots = [np.dot(noisy_hrr.v, h.v) for (k, h) in hrrs]
            sim = max(sims)
            dot = max(dots)

            print "Similarity of closest incorrect index vector ", sim
            print "Dot product of closest incorrect index vector ", dot

            self.second_dot = dot

    def get_similarities_random(self, s, n, dataFunc=None):
        samples_per_vec = 500
        i = 0
        print "In get_similarities_random"
        for idkey1 in self.hrr_vecs.keys():

            key_sample = random.sample(self.hrr_vecs, samples_per_vec)
            vec1 = self.hrr_vecs[idkey1]

            if i % 100 == 0:
                print "Sampling for vector: ", i

            for idkey2 in key_sample:

                if idkey1 == idkey2:
                    continue

                vec2 = self.hrr_vecs[idkey2]

                similarity = vec1.compare(vec2)
                self.add_data(idkey1, similarity)
                self.add_data(idkey2, similarity)
                self.add_data("all", similarity)

            i += 1

    def get_similarities_sample(self, s, n, dataFunc=None):
        num_samples = 2 * len(self.hrr_vecs)
        threshold = 0.1
        print "In get_similarities_sample"
        print "Num samples:" + str(num_samples)
        print "Threshold:" + str(threshold)

        for i in range(num_samples):
            idkey1, idkey2 = random.sample(self.hrr_vecs, 2)
            vec1 = self.hrr_vecs[idkey1]
            vec2 = self.hrr_vecs[idkey2]

            similarity = vec1.compare(vec2)

            if similarity > threshold:
                self.add_data("all", similarity)

            if i % 1000 == 0:
                print "Trial: ", i

    def get_similarities(self, s, n, dataFunc=None):
        """get similarities of idvectors"""
        remaining_keys = self.hrr_vecs.keys()
        for idkey1 in self.hrr_vecs.keys():
            vec1 = self.hrr_vecs[idkey1]

            for idkey2 in remaining_keys:

                if idkey1 == idkey2:
                    continue

                vec2 = self.hrr_vecs[idkey2]

                similarity = vec1.compare(vec2)
                self.add_data(idkey1, similarity)
                self.add_data(idkey2, similarity)
                self.add_data("all", similarity)

            remaining_keys.remove(idkey1)

    def print_config(self, output_file):
        output_file.write("Extractor config:\n")
        output_file.write("Num items: " + str(self.num_items) + "\n")
        output_file.write("Dimension: " + str(self.dimension) + "\n")
        output_file.write("Threshold: " + str(self.threshold) + "\n")
