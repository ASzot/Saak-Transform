from torchvision import datasets, transforms
import torch.multiprocessing
import torch.utils.data as data_utils
from data.datasets import DatasetFromHdf5
from sklearn.cluster import KMeans
import numpy as np
import PIL.Image as Image
import os

import saak

## TO-DO: Perform Kmeans using Saak
# 1. prepare and format data like MNIST
# 2. refer the test case in saak.py to extract saak feature, the desired dimension is [num_images, final_feat_dim]
# 3. perform k-means clustering patches using the saak feature. Try K=34.
# 4. Visualize sample patches of each cluster

cluster_path = '/media/eeb435/media/Junting/data/Project/Cityscapes/saak_patches/32x32/k-means/Clusters'
centroid_path = '/media/eeb435/media/Junting/data/Project/Cityscapes/saak_patches/32x32/k-means/Centroids'

def k_mean_clustering(data = None, feature = None, K = 34, num_centroids_to_visualize = 20):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(feature)
    pred_label = kmeans.labels_
    centroid = kmeans.cluster_centers_
    
    if num_centroids_to_visualize is not -1:
        
        for i in range(K):
            min_f = np.zeros(num_centroids_to_visualize, np.uint8)
            min_distance =np.zeros(num_centroids_to_visualize, np.float32)
            
            for f in range(feature.shape[0]):
                curr_distance = np.linalg.norm(feature[f, :] - centroid[i, :])
                if f < num_centroids_to_visualize:
                    min_distance[f] = curr_distance
                    min_f[f] = f
                elif curr_distance < np.amax(min_distance):
                    max_idx = np.argmax(min_distance)
                    min_distance[max_idx] = curr_distance
                    min_f[max_idx] = f

            for j in range(num_centroids_to_visualize):
                img_array = np.transpose(data[min_f[j], :, :, :], (1, 2, 0))
                img = Image.fromarray(img_array)
                centroid_true_path = centroid_path + '/centroid_' + str(num_centroids_to_visualize) + '/cluster_' + str(i)
                if not os.path.exists(centroid_true_path):
                    os.makedirs(centroid_true_path)
                img.save(centroid_true_path + '/' + str(j) + '.png')
    else:
        
        for i in range(K):
            cluster_data  = data[pred_label==i]
            
            for j in range(cluster_data.shape[0]):
                img_array = np.transpose(cluster_data[j, :, :, :], (1, 2, 0))
                img = Image.fromarray(img_array)

                if not os.path.exists(cluster_path + str(i)):
                    os.makedirs(cluster_path + str(i))
                img.save(cluster_path + str(i) + '/' + str(j) + '.png')
                
if __name__=='__main__':
    # hdf5 file is generated using data/create_hdf5.py
    # using create_hdf5.py, simply run: python create_hdf5.py
    train_path = '/media/eeb435/media/Junting/data/Project/Cityscapes/saak_patches/32x32/hdf5/train_CS_DS2.hdf5'
    # val_path = '/home/chen/dataset_DA/Cityscapes/saak_patches/images/hdf5/val_CS_DS4.hdf5'
    batch_size = 1
    test_batch_size = 1
    kwargs = {}
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_set = DatasetFromHdf5(train_path)
    # val_set = DatasetFromHdf5(val_path)

    train_loader = data_utils.DataLoader(dataset=train_set, num_workers=8,
                                         batch_size=batch_size, shuffle=True, **kwargs)

    # validation_data_loader = data_utils.DataLoader(dataset=val_set, num_workers=8,
    #                                                batch_size=batch_size, shuffle=True, **kwargs)
    K = 3
    NUM_VIS = 15
    NUM_IMAGES = 1000
    num_images = NUM_IMAGES
    data = saak.create_numpy_dataset(num_images, train_loader)
    filters, outputs = saak.multi_stage_saak_trans(data, energy_thresh=0.97)
    final_feat_dim = sum(
        [((output.data.shape[1] - 1) / 2 + 1) * output.data.shape[2] * output.data.shape[3] for output in outputs])
    print 'final feature dimension is {}'.format(final_feat_dim)
    final_feat = saak.get_final_feature(outputs)
    assert final_feat.shape[1] == final_feat_dim
    print(final_feat.shape)
    k_mean_clustering(data=data, feature=final_feat, K=K, num_centroids_to_visualize=NUM_VIS)

