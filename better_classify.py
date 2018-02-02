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

from classify import load_toy_dataset, f_test, svm_classifier, f_test, reduce_feat_dim

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from classify import create_numpy_dataset, get_data_loaders, train_data

from itertools import groupby

import clustering_helper as ch

plt.switch_backend('agg')


def gcm_test(feat, labels):
    # Get average for each class.
    binned = ch.bin_samples(feat, labels)
    print(feat.shape)

    class_mean = list(map(lambda val: (val[0], np.mean(np.array(val[1]).T, axis=1)), binned.items()))

    # [a][b] is the same as [b][a]
    pairwise_distances = {}

    def format_key(c, other_c):
        return str(c) + '_' + str(other_c)

    for c, m in class_mean:
        for other_c, other_m in class_mean:
            if other_c == c:
                continue
            # If the other way already exists
            if format_key(c, other_c) in pairwise_distances or format_key(other_c, c) in pairwise_distances:
                continue
            pairwise_distances[format_key(c, other_c)] = abs(m - other_m)

    ratios = []
    for i in range(feat.shape[1]):
        distances = []
        for class_key in pairwise_distances:
            distances.append(pairwise_distances[class_key][i])

        # For class c_0 and component i across all other classes
        inter_class_dists = sorted(distances, reverse=True)

        gapped = inter_class_dists[0]
        others = inter_class_dists[1:]

        ratio = gapped / np.mean(others)

        ratios.append(ratio)

    ratios = np.array(ratios)
    # Select top 1,000 ratios

    TAKE_COUNT = 1000
    idx = np.argsort(ratios)[:TAKE_COUNT]

    return ratios[idx], idx


def train_data(data, labels):
    data = data.reshape(-1, 3, 32, 32)
    filters, means, outputs = saak.multi_stage_saak_trans(data, energy_thresh=0.97)
    final_feat_dim = sum(
        [((output.shape[1] - 1) / 2 + 1) * output.shape[2] * output.shape[3] for output in outputs])
    # This is the dimensionality of each datapoint.
    print('final feature dimension is {}'.format(final_feat_dim))
    final_feat = saak.get_final_feature(outputs)
    assert final_feat.shape[1] == final_feat_dim

    # Remove some of the features with an f-test
    gcm_test(final_feat, labels)
    selected_feat, idx = f_test(final_feat, labels, thresh=0.75)

    reduced_feat, pca = reduce_feat_dim(selected_feat, dim=248)

    #clf = svm_classifier(reduced_feat, labels)
    clf = svm_classifier(reduced_feat, labels)
    pred = clf.predict(reduced_feat)

    pred_prob = clf.predict_proba(reduced_feat)
    print(pred_prob[0])

    acc = sklearn.metrics.accuracy_score(labels, pred)
    print('training acc is {}'.format(acc))

    return clf, filters, means, final_feat_dim, idx, pca


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader, test_loader = get_data_loaders()

    NUM_IMAGES_TRAIN = 2000
    data, labels = create_numpy_dataset(NUM_IMAGES_TRAIN, train_loader)

    clf, filters, means, final_feat_dim, idx, pca = train_data(data, labels)

    print('\n-----------------start testing-------------\n')

    def create_test_dataset():
        NUM_IMAGES_TEST = 500
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
