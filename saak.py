'''
@ author: Jiali Duan
@ function: Saak Transform
@ Date: 10/29/2017
@ To do: parallelization
'''

# load libs
import torch
import argparse
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from data.datasets import DatasetFromHdf5
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
from cluster_patch import k_mean_clustering

# argument parsing
print torch.__version__
batch_size=1
test_batch_size=1
kwargs={}
# hdf5 file is generated using data/create_hdf5.py
# using create_hdf5.py, simply run: python create_hdf5.py
train_path = '/home/chen/dataset_DA/Cityscapes/saak_patches/images/hdf5/train_CS_DS4.hdf5'
val_path = '/home/chen/dataset_DA/Cityscapes/saak_patches/images/hdf5/val_CS_DS4.hdf5'
K = 34
NUM_VIS = 20

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

train_set = DatasetFromHdf5(train_path)
val_set = DatasetFromHdf5(val_path)

train_loader = data_utils.DataLoader(dataset=train_set, num_workers=8, 
                                            batch_size=batch_size, shuffle=True, **kwargs)

validation_data_loader = data_utils.DataLoader(dataset=val_set, num_workers=8, 
                                            batch_size=batch_size, shuffle=True, **kwargs)



# show sample
def show_sample(inv):
    inv_img=inv.data.numpy()[0][0]
    plt.imshow(inv_img)
    plt.gray()
    plt.savefig('./image/demo.png')
   # plt.show()

'''
@ For demo use, only extracts the first 1000 samples
'''
def create_numpy_dataset(num_images):
    datasets = []
    count = 0
    for data in train_loader:
        if count == num_images:
            break
        data_numpy = data[0].numpy()
        data_numpy = np.squeeze(data_numpy)
        datasets.append(data_numpy)
        count = count + 1
        print(count)

    datasets = np.array(datasets)
    #datasets=np.expand_dims(datasets,axis=1)
    print 'Numpy dataset shape is {}'.format(datasets.shape)
    return datasets[:num_images]



'''
@ data: flatten patch data: (14*14*60000,1,2,2)
@ return: augmented anchors
'''
def PCA_and_augment(data_in, energy_thresh=1.0):
    # data reshape
    data=np.reshape(data_in,(data_in.shape[0],-1))
    print 'PCA_and_augment: {}'.format(data.shape)
    # patch mean removal
    mean = np.mean(data, axis=1, keepdims=True)
    data_mean_remov = data - mean
    print 'PCA_and_augment meanremove shape: {}'.format(data_mean_remov.shape)

    # PCA, retain all components
    if energy_thresh == 1.0:
        pca = PCA(svd_solver='full')
    else:
        pca=PCA(n_components=energy_thresh, svd_solver='full')
    ## if 0 < n_components < 1 and svd_solver == 'full'
    # select the number of components such that the amount of variance that needs to be explained
    # is greater than the percentage specified by n_components

    pca.fit(data_mean_remov)
    comps=pca.components_

    # augment
    comps_neg=[vec*(-1) for vec in comps[:-1]] #the last comp is dc anchor vec, don't augment
    comps_complete=np.vstack((comps, comps_neg))
    print 'PCA_and_augment comps_complete shape: {}'.format(comps_complete.shape)
    return comps_complete



'''
@ datasets: numpy data as input
@ depth: determine shape, initial: 0
'''

def fit_pca_shape(datasets,depth):
    factor=np.power(2,depth)
    length=32/factor
    print 'fit_pca_shape: length: {}'.format(length)
    idx1=range(0,length,2)
    idx2=[i+2 for i in idx1]
    print 'fit_pca_shape: idx1: {}'.format(idx1)
    data_lattice=[datasets[:,:,i:j,k:l] for ((i,j),(k,l)) in product(zip(idx1,idx2),zip(idx1,idx2))]
    data_lattice=np.array(data_lattice)
    print 'fit_pca_shape: data_lattice.shape: {}'.format(data_lattice.shape)
    data=np.reshape(data_lattice,(data_lattice.shape[0]*data_lattice.shape[1],data_lattice.shape[2],2,2))
    print 'fit_pca_shape: reshape: {}'.format(data.shape)
    return data


'''
@ Prepare shape changes. 
@ return filters and datasets for convolution
@ aug_anchors: [7,4] -> [7,input_shape,2,2]
@ output_datas: [60000*num_patch*num_patch,channel,2,2]

'''
def ret_filt_patches(aug_anchors,input_channels):
    shape=aug_anchors.shape[1]/4
    num=aug_anchors.shape[0]
    filt=np.reshape(aug_anchors,(num,shape,4))

    # reshape to kernels, (7,shape,2,2)
    filters=np.reshape(filt,(num,shape,2,2))

    # reshape datasets, (60000*shape*shape,shape,28,28)
    # datasets=np.expand_dims(dataset,axis=1)

    return filters



'''
@ input: numpy kernel and data
@ output: conv+relu result
'''
def conv_and_relu(filters,datasets,stride=2):
    # torch data change
    filters_t=torch.from_numpy(filters)
    datasets_t=torch.from_numpy(datasets)

    # Variables
    filt=Variable(filters_t).type(torch.FloatTensor)
    data=Variable(datasets_t).type(torch.FloatTensor)

    # Convolution
    output=F.conv2d(data,filt,stride=stride)

    # Relu
    relu_output=F.relu(output)

    return relu_output,filt



'''
@ One-stage Saak transform
@ input: datasets [60000, channel, size,size]
'''
def one_stage_saak_trans(datasets=None,depth=0, energy_thresh=1.0):


    # load dataset, (60000,1,32,32)
    # input_channel: 1->7
    print 'one_stage_saak_trans: datasets.shape {}'.format(datasets.shape)
    input_channels=datasets.shape[1]

    # change data shape, (14*60000,4)
    data_flatten=fit_pca_shape(datasets,depth)

    # augmented components
    comps_complete=PCA_and_augment(data_flatten, energy_thresh=energy_thresh)
    print 'one_stage_saak_trans: comps_complete: {}'.format(comps_complete.shape)

    # get filter and datas, (7,1,2,2) (60000,1,32,32)
    filters=ret_filt_patches(comps_complete,input_channels)
    print 'one_stage_saak_trans: filters: {}'.format(filters.shape)

    # output (60000,7,14,14)
    relu_output,filt=conv_and_relu(filters,datasets,stride=2)

    data=relu_output.data.numpy()
    print 'one_stage_saak_trans: output: {}'.format(data.shape)
    return data,filt,relu_output



'''
@ Multi-stage Saak transform
'''
def multi_stage_saak_trans(data, energy_thresh=1.0):
    filters = []
    outputs = []
    num_stages=0
    img_len=data.shape[0]
    
    while(img_len>=2):
        if num_stages >= 5:
            break
        num_stages+=1
        img_len/=2

    print(num_stages)

    for i in range(num_stages):
        print '{} stage of saak transform: '.format(i)
        data,filt,output=one_stage_saak_trans(data, depth=i, energy_thresh=energy_thresh)
        filters.append(filt)
        outputs.append(output)
        print ''


    return filters, outputs

'''
@ Reconstruction from the second last stage
@ In fact, reconstruction can be done from any stage
'''
def toy_recon(outputs,filters):
    outputs=outputs[::-1][1:]
    filters=filters[::-1][1:]
    num=len(outputs)
    data=outputs[0]
    for i in range(num):
        data = F.conv_transpose2d(data, filters[i], stride=2)

    return data

'''
P/S conversion to get useful feature
'''
def p_s_conversion(position_feature):
    n, c, h, w = position_feature.shape
    ac_dim = (c - 1)/2
    dc_feat = position_feature[:, -1:, :, :]
    signed_ac_feat = position_feature[:, :ac_dim, :, :] - position_feature[:, ac_dim:-1, :, :]
    signed_feat = np.concatenate((dc_feat, signed_ac_feat), axis=1)
    return signed_feat

'''
flatten and concat saak features from all stages (stage 1-5)
input: list of output cuboids of multi-stage saak transform
'''
def get_final_feature(outputs):
    final_feat = None
    for output in outputs:
        npy = output.data.numpy()
        signed_output = p_s_conversion(npy)
        flattened_feat = np.reshape(signed_output, [signed_output.shape[0], -1])
        if final_feat is None:
            final_feat = flattened_feat
        else:
            final_feat = np.concatenate((final_feat, flattened_feat), axis=1)
    return final_feat


if __name__=='__main__':
    # Testing
    num_images = 10000
    data = create_numpy_dataset(num_images)
    filters, outputs = multi_stage_saak_trans(data, energy_thresh=0.97)
    final_feat_dim = sum([((output.data.shape[1]-1)/2+1)*output.data.shape[2]*output.data.shape[3] for output in outputs])
    print 'final feature dimension is {}'.format(final_feat_dim)
    final_feat = get_final_feature(outputs)
    assert final_feat.shape[1] == final_feat_dim
    print(final_feat.shape)
    k_mean_clustering(data = data, feature = final_feat, K = K, num_centroids_to_visualize = NUM_VIS)







