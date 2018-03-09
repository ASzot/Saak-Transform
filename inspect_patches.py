from torchvision import datasets, transforms
import torch.multiprocessing
import torch.utils.data as data_utils

import saak
from investigate_coeffs import cifar_labels
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.cluster import MiniBatchKMeans
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

plt.switch_backend('agg')


# feat: [N, C, W, H] array where N is # of samples, C color component.
def extract_patches(feat, stride_w, stride_h, patch_width, patch_height):
    _, _, img_width, img_height = feat.shape

    end_h = patch_height

    patches = []

    while end_h <= img_height:
        end_w = patch_width
        while end_w <= img_width:
            patches.append(feat[:, :, end_w - patch_width:end_w, end_h -
                patch_height:end_h])
            end_w += stride_w

        end_h += stride_h

    return np.array(patches)

def find_least_entropy_patches(patches, class_i):
    TAKE_COUNT = 10
    entropies = [stats.entropy(patch) for patch in patches]

    sorted_indices = np.argsort(entropies)

    entropies = np.array(entropies)

    print('Class %i' % class_i)
    print(stats.describe(entropies))
    print('')

    return sorted_indices[:TAKE_COUNT]


def draw_heat_map(selected_indices, class_i, stride_h, stride_w, patch_width,
        patch_height, img_width, img_height):

    total = np.zeros((img_width, img_height))

    width_count = int((img_width - patch_width) / stride_w)

    for selected_index in selected_indices:

        height_index = int(selected_index / width_count)
        width_index = selected_index % width_count

        total[width_index:width_index+patch_width,
                height_index:height_index+patch_height] = 1.0

    plt.imshow(total, cmap='hot')
    plt.savefig('data/results/patches/' + str(class_i) + '.png')
    plt.clf()


"""
patches: [patch count, N, C, patch width, patch height

Returns:
    The flattened version of the patches where now the each element corresponds
    to a single patch sample [C, patch W, patch H] in flattened form. The
    labels have also been extended to be the class label for each of the
    patches
"""
def get_labels_for_patches(patches, labels):
    patch_count, N, C, patch_width, patch_height = patches.shape

    patches = patches.reshape(N, patch_count, C, patch_width, patch_height)

    patch_labels = []
    for label in labels:
        patch_labels.extend([label for i in range(patch_count)])

    patch_data = patches.reshape(N, patch_count, C * patch_width * patch_height)
    patch_data = patch_data.reshape(N * patch_count, C * patch_width *
            patch_height)

    patch_data, patch_labels = sklearn.utils.shuffle(patch_data, patch_labels)

    return patch_data, patch_labels


def patch_method(feat, labels):
    base_path = 'data/results/patches/'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    # Important Notes
    # - Flattening the number of samples, color channel, width, height

    stride_h = 4
    stride_w = 4
    patch_width = 8
    patch_height = 8

    patches = extract_patches(feat, stride_h, stride_w, patch_width, patch_height)

    patch_data, patch_labels = get_labels_for_patches()

    base_path = 'data/results/patch_clusters/'
    print('Deleting everything in ' + base_path)
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    for cluster_count in [512, 1024]:
        print('Fitting MBK')
        mbk = MiniBatchKMeans(n_clusters=cluster_count, init='k-means++')
        print('Done fitting MBK')
        preds = mbk.fit_predict(patch_data)
        binned_labels = ch.bin_labels(patch_labels, preds)
        bin_probs, totals = ch.convert_bins_to_probs(binned_labels)

        binned_samples = ch.bin_samples(patch_data, preds)
        entropies = ch.bin_entropies(binned_samples)

        max_probs = []
        colors = []
        color_map = {
                0: 'b',
                1: 'g',
                2: 'r',
                3: 'c',
                4: 'm',
                5: 'y',
                6: 'k',
                7: 'w',
                8: 'pink',
                9: 'saddlebrown',
            }
        for b in bin_probs:
            max_prob_ele = bin_probs[b][0]
            max_probs.append(max_prob_ele[1])
            colors.append(color_map[max_prob_ele[0]])

        plt.title('Max Class Probability per Cluster (%i)' % cluster_count)
        plt.bar(np.arange(len(max_probs)), max_probs, color=colors)
        plt.savefig(base_path + 'patch_cluster_hist%i.png' % cluster_count)
        plt.clf()

        plt.title('Entropies per Cluster (%i)' % cluster_count)
        plt.bar(np.arange(len(entropies)), entropies)
        plt.savefig(base_path + 'patch_entropies%i.png' %
                cluster_count)
        plt.clf()

        plt.title('Count per patch (%i)' % cluster_count)
        plt.bar(np.arange(len(totals)), totals)
        plt.savefig(base_path + 'patch_counts%i.png' % cluster_count)
        plt.clf()
        print('Saved all!')


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader, test_loader = get_data_loaders()

    #NUM_IMAGES_TRAIN = 2000
    NUM_IMAGES_TRAIN = 10000
    data, labels = create_numpy_dataset(NUM_IMAGES_TRAIN, train_loader)

    patch_method(data, labels)


if __name__ == '__main__':
    main()
