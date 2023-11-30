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


# Load the pre-trained models
kmeans_PCA_model = joblib.load('kmeans_PCA_model.joblib')
birch_PCA_model = joblib.load('birch_PCA_model.joblib')
# Load the pre-trained models
kmeans_TSNE_model = joblib.load('kmeans_TSNE_model.joblib')
birch_TSNE_model = joblib.load('birch_TSNE_model.joblib')
# Load the pre-trained models
kmeans_Umap_model = joblib.load('kmeans_Umap_model.joblib')
birch_Umap_model = joblib.load('birch_Umap_model.joblib')


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
            kmeans_PCA_model = joblib.load('experiments\kmeans_PCA_model.joblib')
        elif cluster_method == 'Birch':
            birch_PCA_model = joblib.load('experiments\\birch_PCA_model.joblib')
            

    elif red_method == 'TSNE':
        model = TSNE(n_components=p, method='exact', random_state=42)
        red_mat = model.fit_transform(mat)
        if cluster_method == 'Kmeans':
            kmeans_TSNE_model = joblib.load('experiments\kmeans_TSNE_model.joblib')
        elif cluster_method == 'Birch':
            birch_TSNE_model = joblib.load('experiments\\birch_TSNE_model.joblib')

    elif red_method == 'UMAP':
        model = UMAP(n_components=p)
        red_mat = model.fit_transform(mat)
        if cluster_method == 'Kmeans':
            kmeans_Umap_model = joblib.load('experiments\kmeans_Umap_model.joblib')
        elif cluster_method == 'Birch':
            birch_Umap_model = joblib.load('experiments\\birch_Umap_model.joblib')

    else:
        raise Exception("Please select one of the three methods: APC, AFC, UMAP")

    return red_mat
    

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Perform dimensionality reduction and clustering for each method
clusts = ['KMeans', "Birch"]
methods = ['ACP', 'TSNE', 'UMAP']

red_method = input("Give the reduction method chosen : ")
cluster_method = input("Give the reduction method chosen : ")

predict = model_clustering(embeddings, 20, red_method, cluster_method)

