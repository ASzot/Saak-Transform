import numpy as np
import h5py
import os
import glob
from PIL import Image
import cv2


train_addrs = '/home/eeb435/Users/Junting/DA/CycleGAN-tensorflow/datasets/mnist2svhn/trainB/'
train_labels = '/home/eeb435/Users/Junting/DA/CycleGAN-tensorflow/datasets/mnist2svhn/svhn_train_labels.txt'
val_addrs = '/home/eeb435/Users/Junting/DA/CycleGAN-tensorflow/datasets/mnist2svhn/testB/'
val_labels = '/home/eeb435/Users/Junting/DA/CycleGAN-tensorflow/datasets/mnist2svhn/svhn_test_labels.txt'
train_path = '/media/eeb435/media/Junting/data/saak_da/data/svhn_train_full_hwc.hdf5'
val_path = '/media/eeb435/media/Junting/data/saak_da/data/svhn_test_full_hwc.hdf5'
SELECT_IMG_LIST = '/home/eeb435/Users/Junting/DA/CycleGAN-tensorflow/datasets/mnist2svhn/svhn_train.txt'
SELECT_VAL_LIST = '/home/eeb435/Users/Junting/DA/CycleGAN-tensorflow/datasets/mnist2svhn/svhn_test.txt'
NUM_IMG = 73257
NUM_VAL = 26032
data_order = 'hwc'#'chw'
channel = 3

def create_hdf5(hdf5_train_file = None, hdf5_val_file = None, len_train = 0, len_val = 0):

    train_shape = (len_train, 32, 32, 3)
    val_shape = (len_val,  32, 32, 3)

    # open a hdf5 file and create arrays
    if hdf5_train_file is not None:
        hdf5_train_file.create_dataset("img", train_shape, np.int8)
        hdf5_train_file.create_dataset("mean", (channel,), np.float32)
        hdf5_train_file.create_dataset("label", (len_train,), np.int8)
    if hdf5_val_file is not None:
        hdf5_val_file.create_dataset("img", val_shape, np.int8)
        hdf5_val_file.create_dataset("label", (len_val,), np.int8)
        # hdf5_file["val_labels"][...] = val_labels


def img2hdf5(hdf5_train_file = None, hdf5_val_file = None, len_train = 0):
    train_shape = (len_train, 32, 32, 3)


    if hdf5_train_file is not None:
        train_list = open(SELECT_IMG_LIST, 'rb').read().splitlines()
        if train_labels is not None:
            labels = dict()
            with open(train_labels, 'rb') as tl:
                lines = tl.readlines()
            for line in lines:
                [img_name, label] = line.split(' ')
                labels[img_name] = int(label)
        count = 0
        for i in range(len(train_list)):
            for filename in glob.glob(os.path.join(train_addrs,train_list[i])):
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if data_order == 'chw':
                    img = np.rollaxis(img, 2)
                hdf5_train_file["img"][count, ...] = img[None]
                if train_labels is not None:
                    hdf5_train_file["label"][count, ...] = labels[train_list[i]]
                count = count + 1
        # save the mean and close the hdf5 file
        if data_order == 'hwc':
            mean = np.mean(hdf5_train_file["img"], axis=(0, 1, 2))
        elif data_order == 'chw':
            mean = np.mean(hdf5_train_file["img"], axis=(0, 2, 3))
        hdf5_train_file["mean"][...] = mean
        hdf5_train_file.close()

    if hdf5_val_file is not None:
        val_list = open(SELECT_VAL_LIST, 'rb').read().splitlines()
        if val_labels is not None:
            labels = dict()
            with open(val_labels, 'rb') as tl:
                lines = tl.readlines()
            for line in lines:
                [img_name, label] = line.split(' ')
                labels[img_name] = int(label)
        count = 0
        for i in range(len(val_list)):
            for filename in glob.glob(os.path.join(val_addrs, val_list[i])):
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if data_order == 'chw':
                    img = np.rollaxis(img, 2)
                hdf5_val_file["img"][count, ...] = img[None]
                if train_labels is not None:
                    hdf5_val_file["label"][count, ...] = labels[val_list[i]]
                count = count + 1
        hdf5_val_file.close()

def main():
    len_train = NUM_IMG
    len_val = NUM_VAL
    if train_addrs is not None:
        hdf5_train_file = h5py.File(train_path, mode='w')
    else:
        hdf5_train_file = None
    if val_addrs is not None:
        hdf5_val_file = h5py.File(val_path, mode='w')
    else:
        hdf5_val_file = None
    create_hdf5(hdf5_train_file, hdf5_val_file, len_train, len_val)
    img2hdf5(hdf5_train_file, hdf5_val_file, len_train)

if __name__=='__main__':
    main()