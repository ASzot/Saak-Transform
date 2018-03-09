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

from saak import PrintHelper

plt.switch_backend('agg')


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


def plot_entropy_hist(entropies, labels, base_path, comp):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    plt.title('Component %i Min is %.4f' % (comp, np.amin(entropies)))
    plt.ylabel('Entropy')
    plt.xlabel('Class')
    plt.bar(np.arange(len(entropies)), entropies, color='r')

    plt.savefig(base_path + str(comp) + '.png')
    plt.clf()


def plot_histo_ready(freqs, bin_edges, kl_div, comp, c, add_str, base_path):
    use_path = base_path + str(comp) + '/' + str(c) + '/'
    if not os.path.exists(use_path):
        os.makedirs(use_path)

    plt.title('KLD: %.4f' % (kl_div))
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.bar(np.arange(len(bin_edges) - 1), freqs, color='r')
    save_path = use_path + add_str + '.png'
    print('Saving ' + save_path)
    plt.savefig(save_path)
    plt.clf()


def kl_test(feat, labels, should_plot=False):
    print('Using KL test to select coeffs')
    binned = ch.bin_samples(feat, labels)

    binned_items = [(c, compute_energy(samples.T)) for c, samples in
            binned.items()]
    # Binned items is a dictionary where key is the class
    # and value is the data belonging to the class in the form of
    # (# channels, # samples)

    comps = []
    is_first = True
    for c, energies in binned_items:
        for i in range(len(energies)):
            if is_first:
                comps.append({})
            comps[i][c] = energies[i]

        is_first = False

    def agg_other_samples(comp, cur_c):
        aggr = []
        for c, samples in comp.items():
            if c != cur_c:
                aggr.extend(samples)

        return np.array(aggr)

    def normalize_hist(dist, eps=1e-4):
        total = np.sum(dist)
        dist = np.array(dist)
        norm = dist / total
        # Remove any very small values
        norm[norm < eps] = eps
        return norm

    # Get the maximum number of samples across each class
    # (If using the full dataset the max should just be equal to the number of
    # samples per class)
    max_sample_count = np.amax([len(samples) for samples in comps[0].values()])
    # Compute the number of bins to use when constructing the histograms

    bin_count = int(np.sqrt(max_sample_count * 9))

    all_data = [sample for comp in comps for samples in comp.values() for sample in
            samples]

    #min_data = np.amin(all_data)
    #max_data = np.amax(all_data)
    #step_size = (max_data - min_data) / bin_count

    #bin_edges = np.arange(min_data, max_data, step_size)

    print('There are %i bins' % bin_count)

    # Initialize components with to an empty array
    kl_comps = [0.0] * len(comps)

    base_path = 'data/results/compare_entropies/'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    comp_dists = []
    for i, comp in tqdm(enumerate(comps)):
        all_kl = []
        use_bins = bin_count
        dists = []
        for c, samples in comp.items():
            # Aggregate samples for every other class
            others = agg_other_samples(comp, c)

            # Get data distributions for both datasets.
            # (numpy auto setting will automatically determine the number of # bins to use)
            this_dist, bin_edges = np.histogram(samples, bins=bin_count)
            other_dist, _ = np.histogram(others, bins=bin_edges)

            norm_this_dist = normalize_hist(this_dist)
            norm_other_dist = normalize_hist(other_dist)

            # Compute the KL divergence between the two distributions.
            kl_div_1 = ch.kl_div(norm_this_dist, norm_other_dist)
            #kl_div_2 = ch.kl_div(norm_other_dist, norm_this_dist)

            kl_div = kl_div_1

            #kl_div = np.abs((kl_div_1 + kl_div_2) / 2.0)

            dists.append((this_dist, other_dist, bin_edges, i, c, kl_div))
            all_kl.append(kl_div)

        comp_dists.append(dists)

        kl_comps[i] = np.amax(all_kl)

    TAKE_COUNT = 2000
    arg_sorted = np.argsort(kl_comps)
    arg_sorted = np.flipud(arg_sorted)
    idx = arg_sorted[:TAKE_COUNT]

    dists = np.array(dists)
    plot_count = 5

    comp_dists = np.array(comp_dists)
    print('Comp dists len ', len(comp_dists))
    plot_dists = comp_dists[arg_sorted][:plot_count]

    #TODO:
    # I messed up the plotting code and accidently deleted it.
    # But I know it's in a previous version of the git history.

    return feat[:, idx], idx



def entropy_test(feat, labels, should_plot=False):
    PrintHelper.print('Using entroy test to select coeffs')
    PrintHelper.print(np.array(labels).shape)
    binned = ch.bin_samples(feat, labels)

    binned_items = [(c, compute_energy(samples.T)) for c, samples in
            binned.items()]
    # Binned items is a dictionary where key is the class
    # and value is the data belonging to the class in the form of
    # (# channels, # samples)

    comps = []
    is_first = True
    for c, energies in binned_items:
        for i in range(len(energies)):
            if is_first:
                comps.append({})
            comps[i][c] = energies[i]

        is_first = False

    def normalize_hist(dist, eps=1e-4):
        total = np.sum(dist)
        dist = np.array(dist)
        norm = dist / total
        # Remove any very small values
        norm[norm < eps] = eps
        return norm

    # Get the maximum number of samples across each class
    # (If using the full dataset the max should just be equal to the number of
    # samples per class)
    max_sample_count = np.amax([len(samples) for samples in comps[0].values()])
    # Compute the number of bins to use when constructing the histograms

    bin_count = int(np.sqrt(max_sample_count * 9))

    all_data = [sample for comp in comps for samples in comp.values() for sample in
            samples]

    PrintHelper.print('There are %i bins' % bin_count)

    entropy_comps = [0.0] * len(comps)

    if should_plot:
        base_path = 'data/results/compare_entropies/'
        if os.path.exists(base_path):
            print('Removing existing results')
            shutil.rmtree(base_path)

    comp_dists = []
    for i, comp in tqdm(enumerate(comps)):
        all_entropy = []
        use_bins = bin_count
        dists = []

        for c, samples in comp.items():
            this_dist, bin_edges = np.histogram(samples, bins='auto')

            norm_this_dist = normalize_hist(this_dist)

            entropy = ch.entropy(norm_this_dist)

            dists.append((this_dist, bin_edges, i, c, entropy))
            all_entropy.append(entropy)

        comp_dists.append(dists)

        entropy_comps[i] = np.amin(all_entropy)

    TAKE_COUNT = 1000
    PrintHelper.print('Selecting %i top coeffs' % (TAKE_COUNT))
    # We want to go from smallest to largest
    arg_sorted = np.argsort(entropy_comps)
    idx = arg_sorted[:TAKE_COUNT]
    PrintHelper.print('Selected')

    dists = np.array(dists)
    plot_count = 10

    comp_dists = np.array(comp_dists)

    top_plot_dists = comp_dists[arg_sorted][:plot_count]
    bottom_plot_dists = comp_dists[arg_sorted][-plot_count:]

    def plot_all(plot_dists, add_path):
        for dist in plot_dists:
            entropies = [entropy for this_dist, bin_edges, i, c, entropy in dist]
            labels = [c for this_dist, bin_edges, i, c, entropy in dist]

            comp_i = dist[0][2]

            plot_entropy_hist(entropies, labels, base_path + add_path + '/', comp_i)

    if should_plot:
        print('Plotting')

        plot_all(top_plot_dists, 'top')
        plot_all(bottom_plot_dists, 'bottom')

    return feat[:, idx], idx


def random_select(data, labels):
    idx = np.arange(data.shape[1])
    np.random.shuffle(idx)
    idx = idx[:2000]
    return data[:, idx], idx


def train_data(data, labels, plot=False, C=3, W=32, H=32):
    data = data.reshape(-1, C, W, H)
    PrintHelper.print('Incoming data shape is %s' % str(data.shape))
    filters, means, outputs = saak.multi_stage_saak_trans(data, energy_thresh=0.97)
    final_feat_dim = sum(
        [((output.shape[1] - 1) / 2 + 1) * output.shape[2] * output.shape[3] for output in outputs])
    # This is the dimensionality of each datapoint.
    final_feat = saak.get_final_feature(outputs)
    #print('final feature dimension is {}'.format(final_feat.shape[1]))
    assert final_feat.shape[1] == final_feat_dim

    # Remove some of the features with an f-test
    selected_feat, idx = entropy_test(final_feat, labels, plot)
    #selected_feat, idx = f_test(final_feat, labels, thresh=0.75)

    reduced_feat, pca = reduce_feat_dim(selected_feat, dim=248)

    clf = svm_classifier(reduced_feat, labels)
    pred = clf.predict(reduced_feat)

    acc = sklearn.metrics.accuracy_score(labels, pred)
    print('training acc is {}'.format(acc))

    return clf, filters, means, final_feat_dim, idx, pca


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader, test_loader = get_data_loaders()

    #NUM_IMAGES_TRAIN = 2000
    NUM_IMAGES_TRAIN = None
    data, labels = create_numpy_dataset(NUM_IMAGES_TRAIN, train_loader)

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
