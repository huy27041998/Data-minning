from sklearn.cluster import KMeans
from LoadData import LoadData
class Train:
    def __init__(self):
        super().__init__()
    def loadData(self):
        l = LoadData()
        self.data = l.loadData('wine-clustering.csv')
    def train(self):
        k = KMeans(n_clusters=3)
        k.fit(self.data['data'])
    def metric(self):
        pass