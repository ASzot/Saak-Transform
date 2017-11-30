import numpy as np
import h5py
import os
import glob
from PIL import Image
import cv2


train_addrs = '/media/eeb435/media/Junting/data/Project/Cityscapes/saak_patches/32x32/img/train/'
val_addrs = None
train_path = '/media/eeb435/media/Junting/data/Project/Cityscapes/saak_patches/32x32/hdf5/train_CS_DS2.hdf5'
val_path = None
SELECT_IMG_LIST = 'train.txt'
SELECT_VAL_LIST = 'val.txt'
NUM_IMG = 500
NUM_VAL = 0
data_order = 'th'

def create_hdf5(hdf5_train_file = None, hdf5_val_file = None, len_train = 0, len_val = 0):

    train_shape = (len_train, 3, 32, 32)
    val_shape = (len_val,  3, 32, 32)

    # open a hdf5 file and create earrays
    if hdf5_train_file is not None:
        hdf5_train_file.create_dataset("img", train_shape, np.int8)
        hdf5_train_file.create_dataset("mean", train_shape[1:], np.float32)
        hdf5_train_file.create_dataset("labels", (len_train,), np.int8)
        # hdf5_file["train_labels"][...] = train_labels
    if hdf5_val_file is not None:
        hdf5_val_file.create_dataset("img", val_shape, np.int8)
        hdf5_val_file.create_dataset("labels", (len_val,), np.int8)
        # hdf5_file["val_labels"][...] = val_labels


def img2hdf5(hdf5_train_file = None, hdf5_val_file = None, len_train = 0):
    train_shape = (len_train, 3, 32, 32)
    mean = np.zeros(train_shape[1:], np.float32)


    if hdf5_train_file is not None:
        train_list = open(SELECT_IMG_LIST, 'rb').read().splitlines()
        count = 0
        for i in range(len(train_list)):
            for filename in glob.glob(train_addrs + train_list[i] + '*.png'):
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if data_order == 'th':
                    img = np.rollaxis(img, 2)
                hdf5_train_file["img"][count, ...] = img[None]
                mean += img / float(len_train)
                count = count + 1
        # save the mean and close the hdf5 file
        hdf5_train_file["mean"][...] = mean
        hdf5_train_file.close()

    if hdf5_val_file is not None:
        val_list = open(SELECT_VAL_LIST, 'rb').read().splitlines()
        count = 0
        for i in range(len(val_list)):
            for filename in glob.glob(val_addrs + val_list[i] + '*.png'):
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if data_order == 'th':
                    img = np.rollaxis(img, 2)
                hdf5_val_file["img"][count, ...] = img[None]
                count = count + 1
        hdf5_val_file.close()

def main():
    len_train = 128 * NUM_IMG
    len_val = 128 * NUM_VAL
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