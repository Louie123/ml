#coding=utf8
import numpy as np
import copy
def kMeans(dataset, k, centroids=None):
    '''
    centroids: centroids of clusters
    cluster_res: each row indicate the cluster num and distance to the centroid of the row index, respectively.
    '''
    result = []
    m, iter_step = np.shape(dataset)[0], 1
    cluster_res = np.mat(np.zeros((m, 2))) 
    is_cluster_changed = True
    while (is_cluster_changed):
        is_cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_idx = -1
            for j in range(k): 
                dist_ji = np.sqrt(sum(np.power(centroids[j,:]- dataset[i,:], 2)))
                if dist_ji < min_dist:  
                    min_dist = dist_ji; 
                    min_idx = j
            if cluster_res[i, 0] != min_idx:
                is_cluster_changed = True
            cluster_res[i,:] = min_idx, min_dist**2
        print "Iter:", iter_step, '\nClusterNO.\tDist\n', cluster_res ,'\n'
        for cent in range(k):
            pts_in_clust = dataset[np.nonzero(cluster_res[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(pts_in_clust, axis = 0)
        iter_step += 1
        result.append(copy.deepcopy(cluster_res[:, 0]))
    return result, centroids