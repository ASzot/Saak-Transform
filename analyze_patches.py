import numpy as np
from scipy import stats
from scipy.fftpack import dct
from sklearn.cluster import MeanShift, MiniBatchKMeans
from sklearn.utils import shuffle
from sklearn import metrics

from collections import defaultdict

def compute_mse(centroids, points, labels):
    centroid_dists = {}
    centroid_counts = {}
    for point, label in zip(points, labels):
        centroid = centroids[label]

        if label not in centroid_dists:
            centroid_dists[label] = 0

        centroid_dists[label] += np.sum(np.power(point - centroid, 2.0))

        if label not in centroid_counts:
            centroid_counts[label] = 0

        centroid_counts[label] += 1

    dists = {}
    for label in centroid_dists:
        dists[label] = centroid_dists[label] / float(centroid_counts[label])

    return dists


def load_analyze():
    all_patches = np.load('data/processed/patches.npy')
    labels = np.load('data/processed/labels.npy')
    pred_labels = np.load('data/processed/pred_labels.npy')
    cluster_centers = np.load('data/processed/centroids.npy')

    label_count = defaultdict(int)
    for label in pred_labels:
        label_count[label] += 1

    counts = list(label_count.values())
    #print(len(counts))

    print(stats.describe(counts))

    print(metrics.accuracy_score(labels, pred_labels))

    #print(pred_labels.shape)

def reshape_patches(patches):
    patches_shape = patches.shape
    flattened_patches = patches.reshape((patches_shape[0],
        patches_shape[1], patches_shape[2] * patches_shape[3]))

    flattened_patches = flattened_patches.T
    flattened_patches = flattened_patches.reshape((-1, patches_shape[1],
        patches_shape[0]))

    return flattened_patches

def analyze_per_patch(patches):
    # Should be in the form (# patches, ...)
    print('-' * 40)
    for i, patch in enumerate(patches):
        desc = stats.describe(patch, axis=None)
        print('%i) %.5f, %.5f' % (i, desc.mean, desc.variance))
    print('-' * 40)

def analyze_spectral(patches):
    # Should be in the form (# patches, ...)

    for i, patch in enumerate(patches):
        variances = []
        for comp in patch:

            desc = stats.describe(comp)
            variances.append(desc.variance)

        var_desc = stats.describe(variances)

        print('%i) %.5f, %.5f, %.5f, %.5f' % (i, var_desc.minmax[0],
            var_desc.minmax[1], var_desc.mean, var_desc.variance))

def bin_labels(labels, pred_labels):
    freqs = {}
    for label, pred_label in zip(labels, pred_labels):
        if pred_label not in freqs:
            freqs[pred_label] = {}
        if label not in freqs[pred_label]:
            freqs[pred_label][label] = 0
        freqs[pred_label][label] += 1

    return freqs

def analyze(patches):
    patches = reshape_patches(patches)

    print(patches.shape)

    #analyze_per_patch(patches)
    #analyze_spectral(e_patches)

    r_a = dct(patches, norm='ortho', axis=-1)
    labels = []
    all_patches = []
    for i, patch in enumerate(r_a):
        labels.extend([i] * len(patch))
        all_patches.extend(patch)

    all_patches = np.array(all_patches)

    ms = MiniBatchKMeans(n_clusters = 64)
    #all_patches, labels = shuffle(all_patches, labels)

    ##take_count = 10000
    ##all_patches = all_patches[:take_count]
    ##labels = labels[:take_count]

    #all_patches = np.array(all_patches)
    #print(all_patches.shape)

    print('Fitting')
    pred_labels = ms.fit_predict(all_patches)
    print('SS')
    print(metrics.silhouette_score(all_patches, pred_labels, metric='euclidean'))
    freqs = bin_labels(labels, pred_labels)
    for label in freqs:
        print(sorted(freqs[label].items(), key=lambda x: x[1]))

    #print(metrics.accuracy_score(labels, pred_labels))

    ## Group samples
    #groups = {}
    #for l, p in zip(pred_labels, all_patches):
    #    if l not in groups:
    #        groups[l] = []
    #    groups[l].append(p)

    #variances = []
    #for label in groups:
    #    group = np.array(groups[label])
    #    variances.append(np.var(group.flatten()))

    #print(stats.describe(variances))

    #np.save('data/processed/patches.npy', np.array(all_patches))
    #np.save('data/processed/labels.npy', np.array(labels))
    #np.save('data/processed/pred_labels.npy', np.array(pred_labels))
    #np.save('data/processed/centroids.npy', np.array(ms.cluster_centers_))

    #labels = np.load('processed/labels.npy')
    #cluster_centers = np.load('processed/centroids.npy')

#    #flattened_patches = flattened_patches[:, 0:1, :]
#
##    for i in range(patches_shape[1]):
##        spectral_component = flattened_patches[:, i, :]
#
#    all_desc = [stats.describe(patch, axis=None) for patch in
#            flattened_patches]
#
#    variances = [desc.variance for desc in all_desc]
#    print('Spectral comp %i: %.5f, %.5f' % (0, np.mean(variances),
#        np.var(variances)))

    #for use_patch in flattened_patches:
    #    #flattened_patches = flattened_patches.reshape(patches_shape[1],
    #    #        patches_shape[0], -1)
    #    print('Calculating stats for %s' % str(use_patch.shape))
    #    use_patch = use_patch.flatten()
    #    all_desc = [stats.describe(patch, axis=None) for patch in use_patch]

    #print_descs(all_desc)


def print_descs(all_desc):
    print('          Mean,     Min,      Max,      Var')
    for i, desc in enumerate(all_desc):
        if i < 10:
            i = ' ' + str(i)
        print('Patch %s: %.5f, %.5f, %.5f, %.5f' % (str(i), desc.mean,
            desc.minmax[1], desc.minmax[0], desc.variance))
