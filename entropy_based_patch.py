from torchvision import datasets, transforms
import torch.multiprocessing
import torch.utils.data as data_utils

import saak
from investigate_coeffs import cifar_labels
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn import svm
import sklearn

import numpy as np
import os
import shutil
from tqdm import tqdm
from itertools import groupby

from classify import load_toy_dataset, f_test, svm_classifier, f_test, reduce_feat_dim
from classify import create_numpy_dataset, get_data_loaders, train_data
import clustering_helper as ch
from inv_final_dist import compute_energy

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from inspect_patches import extract_patches, get_labels_for_patches

import better_classify

from recursive_kmeans import RecursiveKMeans

plt.switch_backend('agg')


def get_clustering(heirch, n_clusters):
    if heirch:
        return RecursiveKMeans(n_clusters)
    else:
        return MiniBatchKMeans(n_clusters=n_clusters, init='k-means++')


def visualize_bin_entropies(sorted_indices, bin_entropies, top_select,
        bottom_select, base_path, name, title):
    bin_entropies = np.array(bin_entropies)

    if (bottom_select + top_select) < len(bin_entropies):
        select_indices = list(sorted_indices[:top_select])
        select_indices.extend(list(sorted_indices[-bottom_select:]))

        vis = bin_entropies[select_indices]
        colors = ['r'] * top_select
        colors.extend(['b'] * bottom_select)
    else:
        vis = bin_entropies[sorted_indices]
        colors = ['r'] * len(bin_entropies)

    plt.title(title)
    plt.bar(np.arange(len(vis)), vis, color=colors)
    plt.savefig(base_path + name)
    plt.clf()


"""
data: [N, C, W, H]
"""
def process(data, labels, base_path, cluster_count, heirch):
    patch_size = 16
    stride = 1

    patches = extract_patches(data, stride, stride, patch_size, patch_size)

    patches, labels = get_labels_for_patches(patches, labels)

    print('Fitting clustering (%s)' % str(patches.shape))
    clustering = get_clustering(heirch=heirch, n_clusters=cluster_count)
    preds = clustering.fit_predict(patches)
    print('Done fitting clustering')

    # Here a bin corresponds to a cluster
    binned_labels = ch.bin_labels(labels, preds)

    index_to_pred = {}
    i = 0
    for b in binned_labels:
        index_to_pred[i] = b
        i += 1

    bin_probs, totals = ch.convert_bins_to_probs(binned_labels)
    binned_samples = ch.bin_samples(patches, preds)
    bin_entropies = ch.bin_entropies(binned_samples)

    binned_samples_labels = ch.bin_samples_labels(patches, preds, labels)

    sorted_indices = np.argsort(bin_entropies)

    if heirch:
        add_str = '_heirch'
    else:
        add_str = ''

    visualize_bin_entropies(sorted_indices,
            bin_entropies, 60, 60, base_path,
            'entropy_vis%i%s.png' % (cluster_count, add_str),
            'Max and Min Entropies (%i)' % cluster_count)

    visualize_bin_entropies(sorted_indices,
            totals, 60, 60, base_path,
            'count_vis%i%s.png' % (cluster_count, add_str),
            'Counts Per Cluster(%i)' % cluster_count)

    max_probs = []
    for cluster_i in bin_probs:
        label_probs = bin_probs[cluster_i]
        max_probs.append(label_probs[0][1])

    visualize_bin_entropies(sorted_indices,
            max_probs, 60, 60, base_path,
            'prob_dist%i%s.png' % (cluster_count, add_str),
            'Max Probability Per Cluster (%i)' % cluster_count)

    # For the top N bins apply the Saak transform
    top_indices = sorted_indices[:10]
    top_bins = []
    for i in top_indices:
        real_i = index_to_pred[i]
        top_bins.append([real_i, *binned_samples_labels[real_i]])

    #for bin_i, b_samples, b_labels in top_bins:
    #    clf, filters, means, final_feat_dim, idx, pca = better_classify.train_data(b_samples, b_labels, C=3, W=patch_size, H=patch_size)



def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader, test_loader = get_data_loaders()

    base_path = 'data/results/entropy_patches/'
    #print('Deleting everything in ' + base_path)
    #if os.path.exists(base_path):
    #    shutil.rmtree(base_path)
    #os.makedirs(base_path)

    #NUM_IMAGES_TRAIN = 2000
    NUM_IMAGES_TRAIN = 4000
    data, labels = create_numpy_dataset(NUM_IMAGES_TRAIN, train_loader)

    #for k in [1024]:
    #    print('')
    #    print('Processing non-heirch for k = %i' % k)
    #    process(data, labels, base_path, k, False)
    #    print('')

    for k in [512, 1024]:
        print('Processing heirch for k = %i' % k)
        process(data, labels, base_path, k, True)


if __name__ == '__main__':
    main()


