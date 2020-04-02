# =============================================================================
# HOMEWORK 7 - CLUSTERING
# CLUSTERING ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# Import the necessary libraries
from sklearn import datasets, model_selection, metrics, cluster, preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# Function that executes both algorithms in a range of formed clusters from 2
# to 10 and then evaluates them using the Silhouette metric. Finally prints the 
# corresponding plots which visualize the clusters created.
def exec_plot_algo(algo, X_scale):
   
    for n_clusters in range(2,11):
    # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if algo == 1:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            if algo == 2:
                clusterer = cluster.SpectralClustering(n_clusters=n_clusters,assign_labels="discretize", random_state=42)
        cluster_labels = clusterer.fit_predict(X_scale)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_scale, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X_scale, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        
        if algo == 1:
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')
        
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        
        if algo == 1:
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),fontsize=14, fontweight='bold')
        else:
            if algo == 2:
                plt.suptitle(("Silhouette analysis for SpectralClustering clustering on sample data "
                          "with n_clusters = %d" % n_clusters),fontsize=14, fontweight='bold')
        plt.show()



# Load the Breast Cancer dataset.
myData = datasets.load_breast_cancer()

# Retrieve the features (indepent variables) and labels (dependent variables)
X = myData.data
#y = myData.target

# Normalization of features' values to [0,1] scale using MinMaxScaler()
minMaxScaler = preprocessing.MinMaxScaler(copy = False)
X_scale = minMaxScaler.fit_transform(X)


#=============================================================================

# USE AT LEAST 2 CLUSTERING ALGORITHMS TO CLASSIFY THE DATA

#=============================================================================

# 1. K-Means Algorithm

# Evaluate how number of clusters affects model's performance
# I'll evaluate the model for number of clusters from 2 to 10
print('\n\nK-Means Algorithm evaluation:')
exec_plot_algo(1, X_scale)

# =============================================================================


# 2. SpectralClustering Algorithm

# Evaluate how number of clusters affects model's performance
# I'll evaluate the model for number of clusters from 2 to 10
print('\n\nSpectralClustering Algorithm evaluation:')
exec_plot_algo(2, X_scale)
