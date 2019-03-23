import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.neighbors import NearestNeighbors as nn
import MLalgorithms.clustering_utility as util
def Kmeans(data, num_clusters = 2, tolerance=0.0001, max_iter = 300, init_seed = None):
    iter_num = 0
    # Number of training data
    n = data.shape[0]
    # Number of features in the data
    c = data.shape[1]
    # Generate random centers, here i use standard devation 
    #and mean to ensure it represents the whole data
    if(init_seed is None):
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        centroids = np.random.randn(num_clusters,c)*std + mean
    elif(init_seed == 'naive_sharding'):
        centroids = util.naive_sharding(data,num_clusters)
    #elif(init_seed == 'k++'):
        # mean = np.mean(data, axis = 0)
        # std = np.std(data, axis = 0)
        # centroids[0] = np.random.randn(num_clusters,c)*std + mean
        # for i in range(data.shape[0]):
        #     distances[:,i] = np.linalg.norm(data - centroids[0], axis=1)
        # np.argmax(distances, axis = 1)

    else:
        centroids = init_seed
        
    # to store old centers        
    old_centroids = np.zeros(centroids.shape) 
    # Store new centers
    new_centroids = deepcopy(centroids) 
    #generate error vector
    error = np.linalg.norm(new_centroids - old_centroids)
    #create clusters array
    clusters = np.zeros(n)
    #create distaces array
    distances = np.zeros((n,num_clusters))
    # When, after an update, the estimate of that center stays the same, exit loop
    while error > tolerance and iter_num < max_iter:
        iter_num +=1
        # Measure the distance to every center
        for i in range(num_clusters):
            distances[:,i] = np.linalg.norm(data - new_centroids[i], axis=1)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)
        old_centroids = deepcopy(new_centroids)
        # Calculate mean for every cluster and update the center
        for i in range(num_clusters):
            if not data[clusters == i].any() and init_seed is None:
                new_centroids[i] = np.random.randn(1,c)*std + mean
                continue
            new_centroids[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(new_centroids - old_centroids)
        
    return new_centroids,clusters,error,iter_num


def SpectralClustering(data, num_clusters=2, affinity='rbf', gamma=1.0, num_neighbors=1):
    if(affinity == 'rbf'):
        sim_matrix = rbf(data,data,gamma)
    elif(affinity == 'knn'):
        nearest_neigbhor = nn(n_neighbors=num_neighbors)
        nearest_neigbhor.fit(data)
        sim_matrix = nearest_neigbhor.kneighbors_graph(data, mode='connectivity').toarray()
        
    deg_matrix = np.diag(np.sum(sim_matrix, axis=1))
    laplace_matrix = deg_matrix - sim_matrix
    asym_laplace_matrix = np.dot(np.linalg.inv(deg_matrix),laplace_matrix) 
    eig_values,eig_vectors = np.linalg.eig(asym_laplace_matrix)
    idx = np.real(eig_values).argsort()[:num_clusters]
    eig_vectors = np.real(eig_vectors[:,idx])
    rows_norm = np.linalg.norm(eig_vectors, axis=1)
    normalized_eig_vectors = (eig_vectors.T / rows_norm).T
    centroids,clusters,error,iter_num = Kmeans(normalized_eig_vectors, num_clusters=num_clusters)
    return normalized_eig_vectors,centroids,clusters