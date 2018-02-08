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
import analyze_patches
import research_tools as rt

def create_numpy_dataset(num_images, train_loader, take_count=-1):
    datasets = []
    labels = []
    if num_images is None:
        num_images = len(train_loader)
    for i, data in enumerate(train_loader):
        data_numpy = data[0].numpy()
        label_numpy = data[1].numpy()
        label_numpy =  np.squeeze(label_numpy)
        data_numpy = np.squeeze(data_numpy)
        datasets.append(data_numpy)
        labels.append(label_numpy)
        if i==(num_images-1):
            break
    datasets = np.array(datasets)
    labels = np.array(labels)

    if len(datasets.shape)==3: # the input image is grayscale image
        datasets = np.expand_dims(datasets, axis=1)

    print('Numpy dataset shape is {}'.format(datasets.shape))
    if take_count != -1:
        return datasets[:take_count], labels[:take_count]
    else:
        return datasets, labels

def f_test(feat, labels, thresh=0.75):
    f_val,p = f_classif(feat,labels)
    low_conf = p>0.05
    f_val[low_conf] = 0
    where_are_NaNs = np.isnan(f_val)
    f_val[where_are_NaNs] = 0
    idx = f_val > np.sort(f_val)[::-1][int(np.count_nonzero(f_val) * thresh) - 1]
    selected_feat = feat[:, idx]
    print('f-test selected feature shape is {}'.format(selected_feat.shape))
    return selected_feat, idx

def reduce_feat_dim(feat, dim=64):
    pca = PCA(svd_solver='full', n_components=dim)
    reduced_feat = pca.fit_transform(feat)
    print('pca reduced feature shape is {}'.format(reduced_feat.shape))
    return reduced_feat, pca


def knn_classifier(feat, y, N):
    clf = KNeighborsClassifier(N, metric='euclidean')
    clf.fit(feat, y)
    return clf

def svm_classifier(feat, y, kernel='rbf'):
    clf = svm.SVC(kernel=kernel)
    print('Fitting data to svm')
    clf.fit(feat, y)
    print('Data fitted')
    return clf

def get_data_loaders():
    batch_size = 1
    test_batch_size = 1
    kwargs = {}

    data_dir = 'cifar'
    # No need to pad for CIFAR need to pad 2 for MNIST
    load_transform = transforms.Compose([transforms.ToTensor()])
    #load_transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])

    mnist_train = datasets.CIFAR10(root='./data/' + data_dir, train=True,
                                 transform=load_transform,
                                 download=True)
    mnist_test = datasets.CIFAR10(root='./data/' + data_dir, train=False,
                                transform=load_transform,
                                download=True)

    train_loader = data_utils.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = data_utils.DataLoader(mnist_test, batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def load_toy_dataset():
    dataset_path = 'data/mcl/'
    data = []
    labels = []

    appropriate_labels = {
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

    for class_name in os.listdir(dataset_path):
        full_path = dataset_path + class_name + '/'
        appropriate_label = appropriate_labels[class_name]
        for f in os.listdir(full_path):
            data.append(imread(full_path + f) / 255.0)
            labels.append(appropriate_label)

    return np.array(data), np.array(labels)

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
    selected_feat, idx = f_test(final_feat, labels, thresh=0.75)
    reduced_feat, pca = reduce_feat_dim(selected_feat, dim=248)

    #clf = svm_classifier(reduced_feat, labels)
    clf = knn_classifier(reduced_feat, labels, 20)
    pred = clf.predict(reduced_feat)

    acc = sklearn.metrics.accuracy_score(labels, pred)
    print('training acc is {}'.format(acc))

    return clf, filters, means, final_feat_dim, idx, pca


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader, test_loader = get_data_loaders()

    NUM_IMAGES = 10000
    num_images = NUM_IMAGES
    data, labels = create_numpy_dataset(num_images, train_loader)
    toy_data, toy_labels = load_toy_dataset()

    ## outputs is the list of saak coefficients for each layer.
    ## Ex: 0: (N, 7, 16, 16)
    ##     1: (N, 55, 8, 8)
    ##     ...

    clf, filters, means, final_feat_dim, idx, pca = train_data(toy_data, toy_labels)

    print('\n-----------------start testing-------------\n')

    def create_test_dataset():
        test_data, test_labels = create_numpy_dataset(None, test_loader)
        test_outputs = saak.test_multi_stage_saak_trans(test_data, means, filters)
        test_final_feat = saak.get_final_feature(test_outputs)
        return test_final_feat, test_labels

    test_final_feat, test_labels = rt.cached_action(create_test_dataset, 'data/transformed/', ['data', 'labels'])

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



