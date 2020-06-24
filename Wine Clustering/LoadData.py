import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
pd.options.display.float_format = "{:.2f}".format


class LoadData:
    def __init__(self):
        super().__init__()

    def loadData(self, filename):
        data = pd.read_csv(filename)

        # standard scaling
        # std_scaler = StandardScaler()
        # min_max_scaler = MinMaxScaler()
        data_cluster = data.copy()
        # data_cluster[data_cluster.columns] = std_scaler.fit_transform(
        #     data_cluster)
        # data_cluster[data_cluster.columns] = min_max_scaler.fit_transform(
        #     data_cluster)
        # # print(data_cluster.mean())
        # # data = data_cluster.to_numpy()
        # # np.savetxt('data.txt', data, fmt='%.1f')
        # # coverience_matrix = np.dot(np.transpose(data),
        # #                            data) / (data.shape[1] - 1)
        # # np.savetxt('matrix.txt', coverience_matrix, fmt='%.1f')
        # # pca
        # pca_2 = PCA(2)
        # pca_2_result = pca_2.fit_transform(data_cluster)
        # print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_2.explained_variance_ratio_)))
        return data_cluster
