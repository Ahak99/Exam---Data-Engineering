from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import Birch, KMeans
import joblib


def model_clustering(mat, p, red_method, cluster_method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if red_method == 'ACP':
        model = PCA(n_components=p)
        red_mat = model.fit_transform(mat)
        if cluster_method == 'Kmeans':
            model = joblib.load('kmeans_PCA_model.joblib')
        elif cluster_method == 'Birch':
            model = joblib.load('birch_PCA_model.joblib')
            

    elif red_method == 'TSNE':
        model = TSNE(n_components=p, method='exact', random_state=42)
        red_mat = model.fit_transform(mat)
        if cluster_method == 'Kmeans':
            model = kmeans_PCA_model
        elif cluster_method == 'Birch':
            model = joblib.load('birch_TSNE_model.joblib')

    elif red_method == 'UMAP':
        model = UMAP(n_components=p)
        red_mat = model.fit_transform(mat)
        if cluster_method == 'Kmeans':
            model = joblib.load('kmeans_Umap_model.joblib')
        elif cluster_method == 'Birch':
            model = joblib.load('birch_Umap_model.joblib')

    else:
        raise Exception("Please select one of the three methods: APC, AFC, UMAP")

    return model
    

# embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Perform dimensionality reduction and clustering for each method
corpus = input("Give the reduction method chosen : ")
embeddings = model.encode(corpus)

red_method = input("Give the reduction method chosen : ")
cluster_method = input("Give the clustering algorithm chosen : ")

predict = model_clustering(embeddings, 20, red_method, cluster_method)

