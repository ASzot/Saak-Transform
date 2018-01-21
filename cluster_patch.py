from torchvision import datasets, transforms
import torch.multiprocessing
import torch.utils.data as data_utils
from data.datasets import DatasetFromHdf5
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
import PIL.Image as Image
import scipy
import os

import saak
import classify

## TO-DO: Perform Kmeans using Saak
# 1. prepare and format data like MNIST
# 2. refer the test case in saak.py to extract saak feature, the desired dimension is [num_images, final_feat_dim]
# 3. perform k-means clustering patches using the saak feature. Try K=34.
# 4. Visualize sample patches of each cluster

cluster_dir = '/media/eeb435/media/Junting/data/Project/Cityscapes/saak_patches/32x32/k-means/Clusters'
centroid_dir = '/media/eeb435/media/Junting/data/saak_da/kmeans/centroids'

def k_mean_clustering(data = None, feature = None, K = 10, num_centroids_to_visualize = 20):
    kmeans = KMeans(n_clusters=K, random_state=0, n_jobs=-1).fit(feature)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    if num_centroids_to_visualize is not -1:

        neigh = NearestNeighbors(num_centroids_to_visualize, metric='euclidean')

        neigh.fit(feature)
        central_samples_indices = neigh.kneighbors(centroids, num_centroids_to_visualize, return_distance=False)

        for i in range(K):
            centroid_path = os.path.join(centroid_dir, 'centroid_' + str(num_centroids_to_visualize),
                                         'cluster_' + str(i))
            if not os.path.exists(centroid_path):
                os.makedirs(centroid_path)

            for j in range(num_centroids_to_visualize):
                img_array = np.transpose(data[central_samples_indices[i][j], :, :, :], (1, 2, 0)) #move channel to last dim
                scipy.misc.toimage(img_array, cmin=0.0, cmax=1.0, channel_axis=2).save(os.path.join(centroid_path, str(j+1) + '.png'))
    else:
        pass

if __name__=='__main__':
    batch_size = 1
    test_batch_size = 1
    kwargs = {}
    torch.multiprocessing.set_sharing_strategy('file_system')

    # hdf5 file is generated using data/create_hdf5.py
    # using create_hdf5.py, simply run: python create_hdf5.py
    # train_path = '/media/eeb435/media/Junting/data/Project/Cityscapes/saak_patches/32x32/hdf5/train_CS_DS2.hdf5'
    # train_path = '/media/eeb435/media/Junting/data/saak_da/data/svhn_train_full.hdf5'
    # val_path = '/home/chen/dataset_DA/Cityscapes/saak_patches/images/hdf5/val_CS_DS4.hdf5'
    # train_set = DatasetFromHdf5(train_path)
    # val_set = DatasetFromHdf5(val_path)
    # train_loader = data_utils.DataLoader(dataset=train_set, num_workers=8,
    #                                      batch_size=batch_size, shuffle=True, **kwargs)
    # validation_data_loader = data_utils.DataLoader(dataset=val_set, num_workers=8,
    #                                                batch_size=batch_size, shuffle=True, **kwargs)

    svhn_train = datasets.SVHN(root='./data/svhn', split='train', transform=transforms.ToTensor(), download=True)
    svhn_test = datasets.SVHN(root='./data/svhn', split='test', transform=transforms.ToTensor(), download=True)
    train_loader = data_utils.DataLoader(svhn_train, batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = data_utils.DataLoader(svhn_test, batch_size=test_batch_size, shuffle=False, **kwargs)

    K = 10
    NUM_VIS = 20
    NUM_IMAGES = None
    num_images = NUM_IMAGES

    data = saak.create_numpy_dataset(num_images, train_loader)

    filters, means, outputs = saak.multi_stage_saak_trans(data, energy_thresh=0.97)
    final_feat_dim = sum(
        [((output.shape[1] - 1) / 2 + 1) * output.shape[2] * output.shape[3] for output in outputs])
    print('final feature dimension is {}'.format(final_feat_dim))
    final_feat = saak.get_final_feature(outputs)
    assert final_feat.shape[1] == final_feat_dim
    print(final_feat.shape)

    k_mean_clustering(data=data, feature=final_feat, K=K, num_centroids_to_visualize=NUM_VIS)

