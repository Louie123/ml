#coding=utf8
'''
    Toolkit for Document Clustering
    Author: Leyi Wang
    Last update: 2017-07-25
'''
import jieba
import sys
import getopt
import numpy as np
import scipy, collections
import matplotlib.pylab as plt
import scipy.cluster.hierarchy as sch
from sklearn.manifold import TSNE
from cluster.SPC import OnePassCluster
from cluster.kmeans import kMeans

def load_file(corpus_in):
    term_set = set()
    doc_terms_list=[]
    fin = open(corpus_in, 'r').read().split('\n')
    for line in fin:
        doc_terms = line.split()
        doc_terms_list.append(doc_terms)
        term_set.update(doc_terms)
    return doc_terms_list, sorted(list(term_set))

def jieba_seg(corpus_in, corpus_out):
    '''
    Tokenizer for chinese corpus
    '''
    fin = open(corpus_in, 'r')
    fout = open(corpus_out, 'w')
    for line in fin:
        seg_list = jieba.cut(line)
        fout.write(" ".join(seg_list) + "\n")

def stat_df_term(term_set, doc_terms_list):
    '''
    df_term is a dict
    '''
    df_term = {}.fromkeys(term_set, 0)
    for doc_terms in doc_terms_list:
        for term in set(doc_terms):
            if df_term.has_key(term):
                df_term[term] += 1
    return df_term

def stat_idf_term(doc_num, df_term):
    '''
    idf_term is a dict
    '''
    idf_term = {}.fromkeys(df_term.keys())
    for term in idf_term:
        idf_term[term] = np.log(float(doc_num/df_term[term]))
    return idf_term

def stat_tf_term(term_set, doc_terms_list):
    '''
    tf_term is a dict
    '''
    tf_term = {}.fromkeys(term_set, 0)
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            if tf_term.has_key(term):
                tf_term[term] += 1
    return tf_term

def build_samps(term_set, doc_terms_list, term_weight='BOOL'):
    docs_vsm = []
    if term_weight == 'TFIDF':
        df_term = stat_df_term(term_set, doc_terms_list)
        doc_num = len(doc_terms_list)
        idf_term = stat_idf_term(doc_num, df_term)
    for doc in doc_terms_list:
        temp_vector = []
        for term in term_set:
            if term_weight == 'TF':
                temp_vector.append(doc.count(term) * 1.0)
            elif term_weight == 'BOOL':
                temp_vector.append(int(bool(doc.count(term))) * 1.0)
            elif term_weight =='TFIDF':
                temp_vector.append(idf_term[term] * doc.count(term) * 1.0)
            else:
                print 'please check the term_weight parameter'
                return
        docs_vsm.append(temp_vector)
    return np.array(docs_vsm)

def plot_with_labels(low_dim_embs, labels, title):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than data"

    plt.title(title)
    plt.xlabel("X-Axis")
    plt.ylabel('Y-Axis')
    plt.xlim((-150, 150))
    plt.ylim((-170, 150))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='left', va='baseline')

def start_demo(docs_vsm, mode='kmeans'):
    if mode == 'hierarchical':
        print ("############## Hierarchical Clustering Algorithm ###############")
        disMat = sch.distance.pdist(docs_vsm, 'cosine') 
        Z=sch.linkage(disMat, method='weighted') #get linkage matrix single complete average weighted centroid
        P=sch.dendrogram(Z)
        cluster= sch.fcluster(Z, t=1, criterion='inconsistent', depth=2)#get the result of hierarchical clustering 
        print "Hierarchy clustering:\n",cluster
        plt.title('The Result of Hierarchical Clustering Algorithm')
        plt.xlabel("Samples")
        plt.ylabel('Distance')
        plt.savefig('hierarchical.png')
        plt.show()

    elif mode == 'kmeans':
        print("############## K-means Algorithm ###############")
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
        low_dim_doc_embs = tsne.fit_transform(docs_vsm)
        centroids = np.array([docs_vsm[1,:], docs_vsm[4,:], docs_vsm[6,:]])
        result = kMeans(docs_vsm, 3, centroids)[0]
        plt.figure(figsize=(6, 5.2))  # in inches
        for idx, labels in enumerate(result):
            title = 'The Clustering Result of K-means Algorithm (step-' + str(idx + 1) + ')'
            plot_with_labels(low_dim_doc_embs, map(int, range(10)), title)
            plt.scatter(low_dim_doc_embs[:, 0], low_dim_doc_embs[:, 1], c=map(int, labels))
            plt.savefig('K-means_' + str(idx + 1) + '.png')
        plt.show()

    elif mode == "SPC":
        clustering = OnePassCluster(vector_list=docs_vsm, t=2.35)
        labels = clustering.print_result()
        plt.figure(figsize=(6, 5.2))  # in inches
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
        low_dim_doc_embs = tsne.fit_transform(docs_vsm)
        title = 'The Result of Single Pass Clustering Algorithm'
        plot_with_labels(low_dim_doc_embs, map(int, range(10)), title)
        plt.scatter(low_dim_doc_embs[:, 0], low_dim_doc_embs[:, 1], c=map(int, labels))
        plt.savefig('spc.png')
        plt.show()

def feature_selection_tf(temp_term_set, doc_terms_list, min_num=1):
    term_set = set()
    term_tf_dic = stat_tf_term(temp_term_set, doc_terms_list)
    for item in temp_term_set:
        if term_tf_dic[item] > min_num:
            term_set.add(item)
    return term_set

if __name__ == '__main__':
    options, args = getopt.getopt(sys.argv[1:], "hc:m:w:", ["help", "corpus=", "model=", "weight="])
    for name, value in options:
        if name in ("-h","--help"):
            print '''
        Usage: Toolkit for clustering
        Options:  -h, --help, display the usage of commands
                  -c, --corpus, the path of corpus
                  -m, --model [kmeans|hierarchical|SPC], unsupervised learning method for document clustering
                  -w, --weight [BOOL|TF|TFIDF],  the weight of text representation
               '''
            sys.exit()
        if name in ("-c","--corpus"):
            corpus = value
        if name in ("-m","--model"):
            model = value
        if name in ("-w","--weight"):
            weight = value
    doc_terms_list, temp_term_set = load_file(corpus)
    term_set = feature_selection_tf(temp_term_set, doc_terms_list)
    docs_vsm = build_samps(term_set, doc_terms_list, weight)
    start_demo(docs_vsm, model)