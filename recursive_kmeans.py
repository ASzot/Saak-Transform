from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

import clustering_helper as ch
import numpy as np

class CentroidNode(object):
    CUR_LABEL = -1
    def __init__(self, parent, centroid, count):
        self.parent = parent
        self.left = None
        self.right = None
        self.centroid = centroid
        self.count = count
        self.label = CentroidNode.CUR_LABEL
        CentroidNode.CUR_LABEL += 1

class RecursiveKMeans(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.num_levels = np.log2(n_clusters)

    def fit_predict(self, samples):
        CentroidNode.CUR_LABEL = -1
        self.root = CentroidNode(None, None, 0)

        print('Fitting recurisvely')
        self.recur_fit(samples, self.root, 0)

        print('Done fitting recursively')

        #self.print_tree(self.root, 0)

        print('Predicitng..')
        return self.predict(samples)


    def predict(self, samples):
        preds = []
        for sample in samples:
            preds.append(self.recur_predict(sample, self.root))

        return preds


    def recur_predict(self, sample, node):
        if node.left is None and node.right is None:
            return node.label

        left_dist = np.linalg.norm(sample - node.left.centroid)
        right_dist = np.linalg.norm(sample - node.right.centroid)

        if left_dist < right_dist:
            return self.recur_predict(sample, node.left)
        else:
            return self.recur_predict(sample, node.right)

    def print_tree(self, node, depth):
        if node is None:
            return
        print('%i) %i, %i' % (depth, node.label, node.count))
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)

    def recur_fit(self, samples, parent, cur_level):
        mbk = MiniBatchKMeans(n_clusters=2, init='k-means++')

        mbk = mbk.fit(samples)
        centroids = mbk.cluster_centers_
        labels = mbk.predict(samples)

        binned_samples = ch.bin_samples(samples, labels)

        parent.left = CentroidNode(parent, centroids[0],
                len(binned_samples[0]))
        parent.right = CentroidNode(parent, centroids[1],
                len(binned_samples[1]))

        cur_level += 1

        if cur_level < self.num_levels - 1:
            self.recur_fit(binned_samples[0], parent.left, cur_level)
            self.recur_fit(binned_samples[1], parent.right, cur_level)

