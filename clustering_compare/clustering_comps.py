import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import hdbscan
import time
import streamlit as st
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

import requests
import os

DATA_URL = ('https://github.com/theholymath/streamlit-examples/blob/master/clustering_compare/clusterable_data.npy')

if not os.path.exists('clusterable_data.npy'):
    data = requests.get(DATA_URL)
else:
    data = np.load('clusterable_data.npy')

# https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
# data = np.load('clusterable_data.npy')

st.title('Comparing Clustering Algorithms')

if st.checkbox('Show Raw Data'):
    plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
    plt.title('Raw Data')
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    st.pyplot()

st.write('## Plots of Clustering Algorithm')
st.write('***')
def plot_clusters(data, algorithm, args, kwds):


    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)

    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=16)
    plt.text(-0.55, 0.68, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=12)
    st.pyplot()

value = st.sidebar.selectbox("Select Algorithm",["Kmeans", "Affinity Propagation",
                                                 "Mean Shift", "Spectral Clustering",
                                                 "Agglomerative Clustering", "DBScan",
                                                 "HDBScan"])
# st.write(value)


if value == "Kmeans":
    nmbr_clusters = st.sidebar.slider('How many clusters?', 1, 20, 6)
    plot_clusters(data, cluster.KMeans, (), {'n_clusters':nmbr_clusters})
if value == "Affinity Propagation":
    plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})
if value == "Mean Shift":
    plot_clusters(data, cluster.MeanShift, (0.175,), {'cluster_all': False})
if value == "Spectral Clustering":
    nmbr_clusters = st.sidebar.slider('How many clusters?', 1, 20, 6)
    plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters': nmbr_clusters})
if value == "Agglomerative Clustering":
    nmbr_clusters = st.sidebar.slider('How many clusters?', 1, 20, 6)
    linkages = ["ward", "complete", "average", "single"]
    linkage = st.sidebar.selectbox("Select Algorithm", linkages)
    plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters': nmbr_clusters, 'linkage': linkage})
if value == "DBScan":
    epsilon = st.sidebar.slider('Epsilon', 0.005, 0.5, value = 0.025, step=0.005)
    plot_clusters(data, cluster.DBSCAN, (), {'eps': epsilon})
if value == "HDBScan":
    epsilon = st.sidebar.slider('Epsilon', 0.005, 0.5, value = 0.025, step=0.005)
    plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':15})

