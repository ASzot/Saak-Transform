from torchvision import datasets, transforms
import torch.multiprocessing
import torch.utils.data as data_utils
from data.datasets import MNIST, DatasetFromHdf5
import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn import svm
import numpy as np
import PIL.Image as Image
import os

import saak

def create_numpy_dataset(num_images, train_loader):
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
    print 'Numpy dataset shape is {}'.format(datasets.shape)
    return datasets, labels

def f_test(feat, labels, thresh=0.75):
    f_val,p = f_classif(feat,labels)
    low_conf = p>0.05
    f_val[low_conf] = 0
    where_are_NaNs = np.isnan(f_val)
    f_val[where_are_NaNs] = 0
    idx = f_val > np.sort(f_val)[::-1][int(np.count_nonzero(f_val) * thresh) - 1]
    selected_feat = feat[:, idx]
    print 'f-test selected feature shape is {}'.format(selected_feat.shape)
    return selected_feat, idx

def reduce_feat_dim(feat, dim=64):
    pca = PCA(svd_solver='full', n_components=dim)
    reduced_feat = pca.fit_transform(feat)
    print 'pca reduced feature shape is {}'.format(reduced_feat.shape)
    return reduced_feat, pca

def svm_classifier(feat, y, kernel='rbf'):
    clf = svm.SVC(kernel=kernel)
    clf.fit(feat, y)
    return clf





if __name__ == '__main__':
    # hdf5 file is generated using data/create_hdf5.py
    # using create_hdf5.py, simply run: python create_hdf5.py
    # train_path = '/media/eeb435/media/Junting/data/Project/Cityscapes/saak_patches/32x32/hdf5/train_CS_DS2.hdf5'
    train_path = '/media/eeb435/media/Junting/data/saak_da/data/svhn_train_full_hwc.hdf5'
    test_path = '/media/eeb435/media/Junting/data/saak_da/data/svhn_train_full_hwc.hdf5'
    batch_size = 1
    test_batch_size = 1
    kwargs = {}
    torch.multiprocessing.set_sharing_strategy('file_system')

    # # SVHN
    # train_set = DatasetFromHdf5(train_path, transform=transforms.ToTensor()) #transform=transforms.ToTensor()
    # test_set = DatasetFromHdf5(test_path, transform=transforms.ToTensor())
    #
    # train_loader = data_utils.DataLoader(dataset=train_set, num_workers=8,
    #                                      batch_size=batch_size, shuffle=True, **kwargs)
    #
    # test_loader = data_utils.DataLoader(dataset=test_set, num_workers=8,
    #                                                batch_size=batch_size, shuffle=False, **kwargs)

    # MNIST
    train_loader = data_utils.DataLoader(MNIST(root='./data', train=True, process=False, transform=transforms.Compose([
        # transforms.Scale((32, 32)),
        transforms.Pad(2),
        transforms.ToTensor(),
    ])), batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = data_utils.DataLoader(MNIST(root='./data', train=False, process=False, transform=transforms.Compose([
        # transforms.Scale((32, 32)),
        transforms.Pad(2),
        transforms.ToTensor(),
    ])), batch_size=test_batch_size, shuffle=False, **kwargs)

    NUM_IMAGES = 60000
    num_images = NUM_IMAGES
    data, labels = create_numpy_dataset(num_images, train_loader)
    filters, means, outputs = saak.multi_stage_saak_trans(data, energy_thresh=0.97)
    final_feat_dim = sum(
        [((output.shape[1] - 1) / 2 + 1) * output.shape[2] * output.shape[3] for output in outputs])
    print 'final feature dimension is {}'.format(final_feat_dim)
    final_feat = saak.get_final_feature(outputs)
    assert final_feat.shape[1] == final_feat_dim
    selected_feat, idx = f_test(final_feat, labels)
    print selected_feat.shape
    reduced_feat, pca = reduce_feat_dim(selected_feat, dim=64)
    clf = svm_classifier(reduced_feat, labels)
    pred = clf.predict(reduced_feat)
    acc = sklearn.metrics.accuracy_score(labels, pred)
    print 'training acc is {}'.format(acc)

    print '\n-----------------start testing-------------\n'

    test_data, test_labels = create_numpy_dataset(None, test_loader)
    test_outputs = saak.test_multi_stage_saak_trans(test_data, means, filters)
    test_final_feat = saak.get_final_feature(test_outputs)
    assert test_final_feat.shape[1] == final_feat_dim
    print test_final_feat.shape

    test_selected_feat = test_final_feat[:, idx]
    test_reduced_feat = pca.transform(test_selected_feat)
    print 'testing reducued feat shape {}'.format(test_reduced_feat.shape)
    test_pred = clf.predict(test_reduced_feat)
    test_acc = sklearn.metrics.accuracy_score(test_labels, test_pred)
    print 'testing acc is {}'.format(test_acc)








