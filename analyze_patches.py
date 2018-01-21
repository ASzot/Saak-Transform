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

def analyze(patches):
    patches_shape = patches.shape
    flattened_patches = patches.reshape((patches_shape[0],
        patches_shape[1], patches_shape[2] * patches_shape[3]))

    flattened_patches = flattened_patches.T
    flattened_patches = flattened_patches.reshape((-1, patches_shape[0],
        patches_shape[1]))

    r_a = dct(flattened_patches, norm='ortho', axis=-1)
    all_patches = []
    labels = []
    for i, patch in enumerate(r_a):
        labels.extend([i] * len(patch))
        all_patches.extend(patch)

    ms = MiniBatchKMeans(n_clusters = 64)
    #all_patches, labels = shuffle(all_patches, labels)

    #take_count = 10000
    #all_patches = all_patches[:take_count]
    #labels = labels[:take_count]

    all_patches = np.array(all_patches)
    print(all_patches.shape)

    print('Fitting')
    pred_labels = ms.fit_predict(all_patches)
    print(metrics.accuracy_score(labels, pred_labels))

    # Group samples
    groups = {}
    for l, p in zip(pred_labels, all_patches):
        if l not in groups:
            groups[l] = []
        groups[l].append(p)

    variances = []
    for label in groups:
        group = np.array(groups[label])
        variances.append(np.var(group.flatten()))

    print(stats.describe(variances))


    np.save('data/processed/patches.npy', np.array(all_patches))
    np.save('data/processed/labels.npy', np.array(labels))
    np.save('data/processed/pred_labels.npy', np.array(pred_labels))
    np.save('data/processed/centroids.npy', np.array(ms.cluster_centers_))

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
