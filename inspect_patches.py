from torchvision import datasets, transforms
import torch.multiprocessing
import torch.utils.data as data_utils

import saak
from investigate_coeffs import cifar_labels
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
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


def find_least_entropy_patches(patches):
    TAKE_COUNT = 10
    entropies = [ch.entropy(patch) for patch in patches]

    sorted_indices = np.argsort(entropies)

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
    plt.savefig('data/results/patches' + str(class_i) + '.png')
    plt.clf()



def patch_method(feat, labels):
    # Important Notes
    # - Flattening the number of samples, color channel, width, height

    stride_h = 4
    stride_w = 4
    patch_width = 8
    patch_height = 8

    N, C, W, H = feat.shape

    patches = extract_patches(feat, stride_h, stride_w, patch_width, patch_height)
    patch_count = patches.shape[0]
    patches = patches.reshape(N, patch_count, C, patch_width, patch_height)

    binned = ch.bin_samples(patches, labels)

    for class_label in binned:
        class_patches = binned[class_label]
        class_patches = class_patches.reshape(patch_count, -1)

        selected_indices = find_least_entropy_patches(class_patches)
        draw_heat_map(selected_indices, class_label, stride_h, stride_w, patch_width,
                patch_height, W, H)

    raise ValueError()



def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader, test_loader = get_data_loaders()

    NUM_IMAGES_TRAIN = 2000
    #NUM_IMAGES_TRAIN = None
    data, labels = create_numpy_dataset(NUM_IMAGES_TRAIN, train_loader)

    patch_method(data, labels)

    clf, filters, means, final_feat_dim, idx, pca = train_data(data, labels)

    print('\n-----------------start testing-------------\n')

    def create_test_dataset():
        #NUM_IMAGES_TEST = 500
        NUM_IMAGES_TEST = None
        test_data, test_labels = create_numpy_dataset(NUM_IMAGES_TEST, test_loader)
        test_outputs = saak.test_multi_stage_saak_trans(test_data, means, filters)
        test_final_feat = saak.get_final_feature(test_outputs)
        return test_final_feat, test_labels

    test_final_feat, test_labels = create_test_dataset()
    #test_final_feat, test_labels = rt.cached_action(create_test_dataset, 'data/transformed/', ['data', 'labels'])

    assert test_final_feat.shape[1] == final_feat_dim

    # Select only the features as determined by the original f-score feature
    # selection.
    test_selected_feat = test_final_feat[:, idx]
    test_reduced_feat = pca.transform(test_selected_feat)
    print('testing reducued feat shape {}'.format(test_reduced_feat.shape))

    test_pred = clf.predict(test_reduced_feat)

    test_acc = sklearn.metrics.accuracy_score(test_labels, test_pred)
    print('testing acc is {}'.format(test_acc))


if __name__ == '__main__':
    main()
