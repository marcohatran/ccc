from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from copy import deepcopy
from scipy.sparse.csgraph import connected_components
from collections import OrderedDict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
def distance(vecs):
    vec1 = vecs[0]
    vecAll = vecs[1]
    Dis_matrix = pairwise_distances(vec1,vecAll,metric = 'cosine',n_jobs=1)
    Dis_matrix = Dis_matrix.astype(np.float16)
    return Dis_matrix
def chunks_vec(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n]

def create_vector(content_list):
    vectorizer = TfidfVectorizer(token_pattern = "\S+", min_df = 2)
    vectors = vectorizer.fit_transform(content_list)
    print ("Tf-idf shape: " + str(vectors.shape))
    svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
    svd_vectors = svd.fit_transform(vectors)
    vector_chunks = list(chunks_vec(svd_vectors,1000))
    vector_chunks = [(i,svd_vectors) for i in vector_chunks]
    return vector_chunks

def distance_matrix(vector_chunks):
    Dis_matrix = []
    for vector_chunk in vector_chunks:
        Dis = distance(vector_chunk)
        Dis_matrix.append(Dis)
    Dis_matrix = np.vstack(Dis_matrix)
    return(Dis_matrix)

def dis_matrix(content_list):
    vector = create_vector(content_list)
    Dis_matrix = distance_matrix(vector)
    return Dis_matrix

def graph(Dis_matrix):
    THRESHOLD = 0.1

    graph = deepcopy(Dis_matrix)
    graph[graph <= THRESHOLD] = 2
    graph[graph != 2] = 0
    graph[graph == 2] = 1
    graph = graph.astype(np.int8)
    res = connected_components(graph,directed=False)
    return res

def cluster(res):
    cluster_labels = res[1]
    num_cluster = res[0]
    res_cluster = OrderedDict()
    for i in range(0,len(cluster_labels)):
        # print(i)
        if cluster_labels[i] in res_cluster: 
            res_cluster[cluster_labels[i]].append(i)
        else: 
            res_cluster[cluster_labels[i]] = [i]
    print("------------==============--------------")
    res_cluster = [res_cluster[i] for i in range(0,num_cluster)]
    res_cluster = [sorted(r) for r in res_cluster if len(r) > 1]
    res_cluster.sort(key=len,reverse=True)
    return res_cluster

def ListClusterTexts(articles,articleCentroidIds,x) :
    return [articles[i] for i in articleCentroidIds[x]]

"""concatinates all tokenised article texts in an article cluster into a single pseudo-natural text"""
def ConcatinateClusterTexts(articles,articleCentroidIds,K) :
    clusterText = ''
    for article in ListClusterTexts(articles,articleCentroidIds,K) :
        clusterText+=''.join(article)
    return clusterText
def CreateWordCloud(text) :
    #removes STOPWORDS from the chart to make more readable
    return WordCloud(
                     background_color="white",
                     width=500,
                     height=500                    ).generate(text)