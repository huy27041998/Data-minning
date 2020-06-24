from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from LoadData import LoadData
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Train:
    def __init__(self):
        super().__init__()

    def loadData(self):
        l = LoadData()
        data_cluster = l.loadData('wine-clustering.csv')
        self.std_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        data_cluster[data_cluster.columns] = self.std_scaler.fit_transform(
            data_cluster)
        data_cluster[data_cluster.columns] = self.min_max_scaler.fit_transform(
            data_cluster)
        # print(data_cluster.mean())
        # data = data_cluster.to_numpy()
        # np.savetxt('data.txt', data, fmt='%.1f')
        # coverience_matrix = np.dot(np.transpose(data),
        #                            data) / (data.shape[1] - 1)
        # np.savetxt('matrix.txt', coverience_matrix, fmt='%.1f')
        # pca
        self.pca_2 = PCA(2)
        self.pca_2_result = self.pca_2.fit_transform(data_cluster)
        self.data = data_cluster

    def train(self):
        self.loadData()
        self.inertia = []
        for i in tqdm(range(2, 10)):
            kmeans = cluster.KMeans(n_clusters=i,
                                    n_init=15,
                                    max_iter=500,
                                    random_state=17)
            kmeans.fit(self.data)
            self.inertia.append(kmeans.inertia_)

        self.silhouette = {}
        for i in tqdm(range(2, 10)):
            kmeans = cluster.KMeans(n_clusters=i,
                                    n_init=15,
                                    max_iter=500,
                                    random_state=17)
            kmeans.fit(self.data)
            self.silhouette[i] = silhouette_score(self.data,
                                                  kmeans.labels_,
                                                  metric='euclidean')
        self.metric()

    def metric(self):
        print(self.silhouette)
        sns.set(style='white', font_scale=1.1, rc={'figure.figsize': (12, 5)})
        plt.bar(range(len(self.silhouette)),
                list(self.silhouette.values()),
                align='center',
                color='red',
                width=0.5)
        plt.xticks(range(len(self.silhouette)), list(self.silhouette.keys()))
        plt.grid()
        plt.title('Silhouette Score', fontweight='bold')
        plt.xlabel('Number of Clusters')
        plt.show()

    def train_with_num_cluster(self, num_cluster):
        self.loadData()
        self.kmeans = cluster.KMeans(n_clusters=num_cluster, random_state=17)
        self.kmeans_labels = self.kmeans.fit_predict(self.data)
        pca_2 = PCA(2)
        pca_2_result = pca_2.fit_transform(self.data)
        self.centroids = self.kmeans.cluster_centers_
        self.centroids_pca = pca_2.transform(self.centroids)
        pd.Series(self.kmeans_labels).value_counts()
        self.metric_with_num_cluster(num_cluster)

    def metric_with_num_cluster(self, num_cluster):
        data2 = self.data.copy()
        data2['Cluster'] = self.kmeans_labels
        aux = data2.columns.tolist()
        for cluster in aux[0:len(aux) - 1]:
            grid = sns.FacetGrid(data2, col='Cluster')
            grid.map(plt.hist, cluster, color='red')
        plt.show()
        centroids_data = pd.DataFrame(data=self.std_scaler.inverse_transform(
            self.centroids),
                                      columns=self.data.columns)
        centroids_data.head()
        sns.set(style='white', rc={'figure.figsize': (9, 6)}, font_scale=1.1)

        plt.scatter(x=self.pca_2_result[:, 0],
                    y=self.pca_2_result[:, 1],
                    c=self.kmeans_labels,
                    cmap='autumn')
        plt.scatter(self.centroids_pca[:, 0],
                    self.centroids_pca[:, 1],
                    marker='x',
                    s=169,
                    linewidths=3,
                    color='black',
                    zorder=10,
                    lw=3)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Clustered Data (PCA visualization)', fontweight='bold')
        plt.show()


t = Train()
# t.train()
t.train_with_num_cluster(3)