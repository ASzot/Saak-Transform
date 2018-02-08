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

from inv_final_dist import compute_energy

plt.switch_backend('agg')

def plot_normal(mu, sigma):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma), 'k')


def plot_comps(comps, add_path, max_dists):
    use_path = 'data/results/energy_hists/'

    for i, (comp, max_dist) in enumerate(zip(comps, max_dists)):
        plt.title('Max: ' + str(max_dist))
        for m, v in comp:
            plot_normal(m, np.sqrt(v))

        total_path = use_path + add_path + '/' + str(i) + '/'
        if not os.path.exists(total_path):
            os.makedirs(total_path)

        plt.savefig(total_path + 'total.png')
        plt.clf()



def plot_histo(data, comp_i, class_i, mean, std, add_path):
    use_path = 'data/results/energy_hists/'
    total_path = use_path + add_path + '/' + str(comp_i) + '/'

    if not os.path.exists(total_path):
        os.makedirs(total_path)

    plt.hist(data, 50, facecolor='green')
    plt.title('mu %.4f, STD: %.4f' % (mean, std))
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    save_path = total_path + str(class_i) + '.png'
    print('Saving to ' + save_path)
    plt.savefig(save_path)
    plt.clf()


def plot_dist_hist(comps, add_path):
    for comp_i, comp in enumerate(comps):

        for class_i, data in enumerate(comp):
            plot_histo(data, comp_i, class_i, np.mean(data), np.std(data), add_path)



def gcm_test(feat, labels):
    print(feat.shape)
    # Get average for each class.
    binned = ch.bin_samples(feat, labels)

    binned_items = [(c, compute_energy(samples.T)) for c, samples in
            binned.items()]

    class_mean = [(val[0], np.mean(val[1], axis=1), np.var(val[1], axis=1), val[1])
            for val in binned_items]

    # Get the means of each class for each component.
    comps = []
    first_pass = True
    for c, mean, var, val in class_mean:
        for i, (m, v) in enumerate(zip(mean, var)):
            if first_pass:
                comps.append([(c, m, v, val[1])])
            else:
                comps[i].append((c, m, v, val[1]))

        first_pass = False

    use_index = 1

    comps = list(map(lambda means: sorted(means, key=lambda x: x[use_index]),
        comps))

    disp_comps = [[[comp[1], comp[2]] for comp in c] for c in comps]
    disp_comps = np.array(disp_comps)

    val_comps = [[comp[3] for comp in c] for c in comps]
    val_comps = np.array(val_comps)

    # Select only the mean
    comp_means = [[classes[use_index] for classes in comp] for comp in comps]
    comp_dists = [np.diff(comp) for comp in comp_means]

    # Get the max from each.
    max_dists = np.array([np.amax(comp_dist) for comp_dist in comp_dists])
    print(stats.describe(max_dists))

    TAKE_COUNT = 2000
    arg_sorted = np.argsort(max_dists)
    arg_sorted = np.flipud(arg_sorted)
    idx = arg_sorted[:TAKE_COUNT]

    selected_class_data = disp_comps[arg_sorted]

    use_count = 25

    plot_dist_hist(val_comps[arg_sorted][:use_count], 'tops')
    plot_dist_hist(val_comps[arg_sorted][-use_count:], 'bottoms')
    plot_comps(selected_class_data[:use_count], 'tops', max_dists[arg_sorted][:100])
    plot_comps(selected_class_data[-use_count:], 'bottoms', max_dists[arg_sorted][-100:])
    raise ValueError()

    # Take the same elements across each sample

    #all_idx = []
    #for i in range(feat.shape[0]):
    #    all_idx.append(idx)

    #all_idx = np.array(all_idx)

    return feat[:, idx], idx


def kl_test(data, labels):
    binned = ch.bin_samples(feat, labels)

    binned_items = [(c, compute_energy(samples.T)) for c, samples in
            binned.items()]

    print(len(binned_items))



    return feat[:, idx], idx


def random_select(data, labels):
    idx = np.arange(data.shape[1])
    np.random.shuffle(idx)
    idx = idx[:2000]
    return data[:, idx], idx


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
    print('Using GCM')
    selected_feat, idx = kl_test(final_feat, labels)
    #selected_feat, idx = f_test(final_feat, labels, thresh=0.75)
    print('Selected using gcm')
    print(selected_feat.shape)

    reduced_feat, pca = reduce_feat_dim(selected_feat, dim=248)

    clf = svm_classifier(reduced_feat, labels)
    pred = clf.predict(reduced_feat)

    acc = sklearn.metrics.accuracy_score(labels, pred)
    print('training acc is {}'.format(acc))

    return clf, filters, means, final_feat_dim, idx, pca


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader, test_loader = get_data_loaders()

    NUM_IMAGES_TRAIN = 2000
    #NUM_IMAGES_TRAIN = None
    data, labels = create_numpy_dataset(NUM_IMAGES_TRAIN, train_loader)

    clf, filters, means, final_feat_dim, idx, pca = train_data(data, labels)

    print('\n-----------------start testing-------------\n')

    def create_test_dataset():
        NUM_IMAGES_TEST = 500
        #NUM_IMAGES_TEST = None
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
    #test_prob = clf.predict_proba(test_reduced_feat)

    #wrong_preds_diff = []
    #right_preds_diff = []

    #wrong_preds_var = []
    #right_preds_var = []

    #count_right = 0
    #count_wrong = 0

    #for t_lbl, t_pred, prob in zip(test_labels, test_pred, test_prob):
    #    max_vals = np.sort(prob)[::-1]
    #    prob_diff = max_vals[0] - max_vals[1]
    #    prob_var = np.var(max_vals)
    #    diff_one = 1 - max_vals[0]
    #    if t_pred != t_lbl:
    #        if diff_one < 0.25:
    #            count_right += 1
    #        wrong_preds_diff.append(diff_one)
    #        wrong_preds_var.append(prob_var)
    #    else:
    #        if diff_one < 0.25:
    #            count_wrong += 1
    #        right_preds_diff.append(diff_one)
    #        right_preds_var.append(prob_var)

    ##print(stats.describe(right_preds_var))
    #print(stats.describe(wrong_preds_diff))
    ##print(stats.describe(wrong_preds_var))
    #print(count_right)
    #print(count_wrong)

    test_acc = sklearn.metrics.accuracy_score(test_labels, test_pred)
    print('testing acc is {}'.format(test_acc))


if __name__ == '__main__':
    main()
