from torchvision import datasets, transforms
import torch.multiprocessing
import torch.utils.data as data_utils
from data.datasets import MNIST, DatasetFromHdf5
import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import imread
import numpy as np
import PIL.Image as Image
import os

import saak
import clustering_helper as ch
from classify import get_data_loaders, create_numpy_dataset

from scipy import stats
from sklearn.cluster import MeanShift, MiniBatchKMeans
import matplotlib.pyplot as plt

plt.switch_backend('agg')

cifar_labels = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
            }



def analyze_all_patches(patches, num_patches, sample_count, spectral_count):
    patches = patches.reshape(num_patches * sample_count, spectral_count)
    labels = [lbl for i in range(num_patches) for lbl in [i] * sample_count]
    labels = np.array(labels)

    print(patches.shape)
    print(labels.shape)

    mb = MiniBatchKMeans(n_clusters = 64)
    pred_labels = mb.fit_predict(patches)

    freqs = ch.bin_labels(labels, pred_labels)

    freq_pcts, total_vals = ch.convert_bins_to_pcts(freqs)

    for i in freq_pcts:
        print('Patch %i (%i)' % (i, total_vals[i]))
        print(freq_pcts[i][:3])

def analyze_single_patch(patches, patch_i, patch_j, num_patches, sample_count, spectral_count):
    patches = patches.reshape(8, 8, sample_count, spectral_count)

    patch = patches[patch_i][patch_j]
    #for p in patch:
    #    stats.describe(p).variance
    return stats.describe(patch, axis=None)

def draw_variance_heatmap(class_type, draw, result_folder, prepend_path=''):
    final = np.zeros((8, 8))
    for var, (i, j) in draw:
        final[i][j] = var

    plt.imshow(final, cmap='hot')
    plt.title(class_type)
    start_path = 'data/results/'
    if prepend_path != '':
        start_path += prepend_path
    plt.savefig(start_path + '%s/var_%s.png' % (result_folder, class_type))

def analyze(patches, class_name, prepend_path = ''):
    p_s = patches.shape
    num_patches = p_s[2] * p_s[3]
    sample_count = p_s[0]
    spectral_count = p_s[1]
    patches = patches.reshape(sample_count, p_s[1], num_patches)

    all_desc = []
    for i in range(8):
        for j in range(8):
            desc = analyze_single_patch(patches, i, j, num_patches, sample_count, spectral_count)
            all_desc.append((desc.variance, (i, j)))

    # Get the 10 patches with the max variance
    all_desc = sorted(all_desc, key = lambda x: x[0], reverse=True)
    max_var = all_desc[:10]
    print(class_name)
    ind_desc = stats.describe([var for var, (i, j) in all_desc])
    print('Mean: %.4f, Var: %.4f, Min: %.4f, Max: %.4f' % (ind_desc.mean,
        ind_desc.variance, ind_desc.minmax[0], ind_desc.minmax[1]))
    print('')
    draw_variance_heatmap(class_name, max_var, 'individual', prepend_path)
    draw_variance_heatmap(class_name, all_desc, 'all', prepend_path)

    return max_var

def main(class_name):
    torch.multiprocessing.set_sharing_strategy('file_system')
    e_save_file_path = 'data/processed/e_examine_patches.npy'
    e_data = []

    use_dir = 'data/mcl/' + class_name + '/'
    for f in os.listdir(use_dir):
        e_data.append(imread(use_dir + f) / 255.0)

    e_data = np.array(e_data).reshape(-1, 3, 32, 32)

    e_filters, e_means, e_outputs = saak.multi_stage_saak_trans(e_data, energy_thresh=0.97)

    e_examine_level = e_outputs[1]

    return analyze(e_examine_level, class_name)

def cifar():
    train_loader, test_loader = get_data_loaders()
    NUM_IMAGES = 10000
    data, labels = create_numpy_dataset(NUM_IMAGES, train_loader)
    filters, means, outputs = saak.multi_stage_saak_trans(data, energy_thresh=0.97)
    examine_level = outputs[1]

    for class_name in cifar_labels:
        use_label = cifar_labels[class_name]
        filtered_data = [sample for sample, label in zip(examine_level, labels)
                if label == use_label]

        print('%i in data class %s' % (len(filtered_data), class_name))
        filtered_data = np.array(filtered_data)

        analyze(filtered_data, class_name, 'cifar/')


def analyze_max_vars(max_vars, class_names):
    exists = {}
    for i in range(8):
        for j in range(8):
            if i not in exists:
                exists[i] = {}
            exists[i][j] = False

    all_non_exist = []
    for max_var in max_vars:
        non_exist = []
        for var, (i, j) in max_var:
            if exists[i][j]:
                non_exist.append((var, (i,j)))
                exists[i][j] = True

        all_non_exist.append(non_exist)

    for max_var, class_name in zip(all_non_exist, class_names):
        draw_variance_heatmap(class_name, max_var, 'non_overlapping')



if __name__ == '__main__':
    #max_vars = []
    #class_names = os.listdir('data/mcl/')
    #for class_name in class_names:
    #    max_var = main(class_name)
    #    max_vars.append(max_var)
    ##analyze_max_vars(max_vars, class_names)

    cifar()
