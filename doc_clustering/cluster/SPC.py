# coding=utf-8
import numpy as np

class ClusterUnit:
    '''
    Params:
    node_list: list to save node.
    node_num: number of nodes in the cluster unit.
    Centroid: the centroid of the current cluster.
    '''
    def __init__(self):  
        self.node_list = []
        self.node_num = 0
        self.centroid = None

    def add_node(self, node, node_vec):  
        """ 
         Function: add node to cluster and update the centroid
         Params:
         node_vec: the vector of node;
         node: node to add
        """  
        self.node_list.append(node)  
        try:  
            self.centroid = (self.node_num * self.centroid + node_vec) / (self.node_num + 1)  #updata the centroid
        except TypeError:  
            self.centroid = np.array(node_vec) #Intailize the centroid
        self.node_num += 1
        
class OnePassCluster:
    '''
    Single Pass Cluster
    Params:
    t: the threshold for single pass cluster
    vector_list: data for clustering
    cluster_list:result of clustering
    '''
    def __init__(self, t, vector_list):
        self.threshold = t
        self.vectors = np.array(vector_list)
        self.cluster_list = []
        self.clustering()  

    def clustering(self):  
        self.cluster_list.append(ClusterUnit())#Initialize the first cluster
        self.cluster_list[0].add_node(0, self.vectors[0])#Add the first Node to the cluster
        for idx in range(1, len(self.vectors)):
            min_dist = np.inf     #the distance to the nearest cluster
            min_cluster_idx = -1  # Index of the nearest cluster
            for cluster_idx, cluster in enumerate(self.cluster_list):
                distance = np.sqrt(sum(np.power(self.vectors[idx] - cluster.centroid, 2)))
                if distance < min_dist:
                    min_dist = distance
                    min_cluster_idx = cluster_idx
            if min_dist < self.threshold:  #the distance to the nearest cluster is less equal to the threshold
                self.cluster_list[min_cluster_idx].add_node(idx, self.vectors[idx])
            else:#create a new cluster
                new_cluster = ClusterUnit()  
                new_cluster.add_node(idx, self.vectors[idx])  
                self.cluster_list.append(new_cluster)

    def print_result(self, label_dict=None):  
        print "#################### Single-Pass Clustering ####################"
        res = np.array([-1] * len(self.vectors))
        for idx, cluster in enumerate(self.cluster_list):
            res[cluster.node_list] = np.array([idx]*cluster.node_num  )
            print "Cluster %s " % idx, cluster.node_list
            print "node num: %s" % cluster.node_num  
            print "----------------------------"  
        print "Total number of nodes and clusters: %s, %s" % (len(self.vectors), len(self.cluster_list))
        return res