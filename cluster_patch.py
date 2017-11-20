
from sklearn.cluster import KMeans
import numpy as np
import PIL.Image as Image
import os

## TO-DO: Perform Kmeans using Saak
# 1. prepare and format data like MNIST
# 2. refer the test case in saak.py to extract saak feature, the desired dimension is [num_images, final_feat_dim]
# 3. perform k-means clustering patches using the saak feature. Try K=34.
# 4. Visualize sample patches of each cluster

cluster_path = '/home/chen/dataset_DA/SAAK_test/Cluster'
centroid_path = '/home/chen/dataset_DA/SAAK_test/Centroid'

def k_mean_clustering(data = None, feature = None, K = 34, num_centroids_to_visualize = 20):

	normlized_feature = feature/np.expand_dims(np.var(feature, 1), 1)
	kmeans = KMeans(n_clusters=K, random_state=0).fit(normlized_feature)
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


