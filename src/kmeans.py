from sklearn.model_selection import train_test_split
from sklearn import datasets
from utils import eucledian_distance
from random import randrange
import operator
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


class MyKMeans:
    def __init__(self):
        self.dataset = []
        self.centroids = []
        self.clusters = []
        self.n_cluster = 0

    def init_cluster(self):
        self.centroids = [self.dataset[randrange(
            0, len(self.dataset))] for _ in range(self.n_cluster)]
        self.refresh_cluster()

    def refresh_cluster(self):
        self.clusters = []

        for data in self.dataset:
            cluster = 0
            min_dist = eucledian_distance(data, self.centroids[0])
            for i in range(1, len(self.centroids)):
                if eucledian_distance(data, self.centroids[i]) < min_dist:
                    min_dist = eucledian_distance(data, self.centroids[i])
                    cluster = i

            self.clusters.append(cluster)

    def calculate_mean(self, datas):
        result = datas[0]

        for data_idx in range(1, len(datas)):
            for column_idx in range(len(datas[data_idx])):
                result[column_idx] += datas[data_idx][column_idx]

        for result_idx in range(len(result)):
            result[result_idx] /= len(datas)

        return result

    def recalculate_centroids(self):
        for centroid_idx in range(len(self.centroids)):
            datas = []

            for cluster_idx, cluster in enumerate(self.clusters):
                if cluster == centroid_idx:
                    datas.append(self.dataset[cluster_idx])

            if (len(datas) > 0):
                self.centroids[centroid_idx] = self.calculate_mean(datas)

    def is_stop(self, prev_centroids):
        if len(self.centroids) != len(prev_centroids):
            return False
        for centroid_idx in range(len(self.centroids)):
            if self.centroids[centroid_idx].tolist() != prev_centroids[centroid_idx].tolist():
                return False
        return True

    def fit(self, dataset, n_cluster):
        self.dataset = dataset
        self.n_cluster = n_cluster

        self.init_cluster()
        prev_centroids = []

        while not self.is_stop(prev_centroids):
            prev_centroids = self.centroids[:]
            self.recalculate_centroids()
            self.refresh_cluster()

    def predict(self, test_data):
        result = []

        for td in test_data:
            closest_centroid = 0
            min_distance = eucledian_distance(self.centroids[closest_centroid], td)
            for i in range(1, len(self.centroids)):
                distance = eucledian_distance(self.centroids[i], td)
                if (distance < min_distance):
                    min_distance = distance
                    closest_centroid = i

            result.append(closest_centroid)
        return result

def create_mapper(target, cluster):
    cluster_set = set(cluster)

    map = {}
    for c in cluster_set:
        counter = {}
        for j in range(len(target)):
            if cluster[j] == c:
                value = target[j]
                if value in counter.keys():
                    counter[value] = counter[value] + 1
                else:
                    counter[value] = 1
        max_key = max(counter.items(), key=operator.itemgetter(1))[0]
        map[c] = max_key
    return map

def count_accuracy(predict, y_test):
    correct_count = 0
    for i in range(len(predict)):
        if predict[i] == y_test[i]:
            correct_count = correct_count + 1
    
    return correct_count*100/len(predict)

def scikmeans_predict(scikit_kmeans,test_data ):
    result = []

    for td in test_data:
        closest_centroid = 0
        min_distance = eucledian_distance(scikit_kmeans.cluster_centers_[closest_centroid], td)
        for i in range(1, len(scikit_kmeans.cluster_centers_)):
            distance = eucledian_distance(scikit_kmeans.cluster_centers_[i],td)
            if (distance < min_distance):
                min_distance = distance
                closest_centroid = i

        result.append(closest_centroid)
    return result

if __name__ == "__main__":
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=23 )

    mykmeans = MyKMeans()
    scikitkmeans = KMeans(n_clusters=4)

    mykmeans.fit(X_train, 4)
    scikitkmeans.fit(X_train)

    print("Clustering Result")
    print(mykmeans.clusters)
    print(scikitkmeans.labels_.tolist())

    mykmeans_map = create_mapper(y_train, mykmeans.clusters)
    scikitkmeans_map = create_mapper(y_train, scikitkmeans.labels_)
    print("Mapping Result")
    print(mykmeans_map)
    print(scikitkmeans_map)

    mykmeans_predict = mykmeans.predict(X_test)
    scikitkmeans_predict = scikmeans_predict(scikitkmeans, X_test)

    print("Predicting Result Before Mapping")
    print(mykmeans_predict)
    print(scikitkmeans_predict)
    for i in range(len(mykmeans_predict)):
        mykmeans_predict[i] = mykmeans_map[mykmeans_predict[i]]
        scikitkmeans_predict[i] = scikitkmeans_map[scikitkmeans_predict[i]]

    print("Solution")
    print(y_test)

    print("Predicting Result After Mapping")
    print(mykmeans_predict)
    print(scikitkmeans_predict)

    print("Evaluation")
    print("My KMeans accuracy", count_accuracy(mykmeans_predict,y_test), "%")
    print("Scikit KMeans accuracy", count_accuracy(scikitkmeans_predict,y_test), "%")