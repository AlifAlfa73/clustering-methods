from sklearn import datasets
from utils import eucledian_distance
import collections
import operator
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import numpy as np

# max(stats.iteritems(), key=operator.itemgetter(1))[0]

#DBSCAN Reference : https://github.com/chrisjmccormick/dbscan/blob/master/dbscan.py

class MyDBSCAN:
    def __init__(self):
        self.dataset = None
        self.cluster = None
        self.eps = None
        self.minpts = None

    def search_neighbor(self, data_index):
        neighbors = []
        data = self.dataset[data_index]

        for i in range(len(self.dataset)):
            dist = eucledian_distance(self.dataset[i],data)
            if(dist < self.eps and i!= data_index):
                neighbors.append(i)
        
        return neighbors

    def create_cluster(self, neighbors, c):
        i = 0
        while i < len (neighbors):
            n = neighbors[i]

            if self.cluster[n] == -1:
                self.cluster[n] = c
            elif self.cluster[n] == 0 :
                self.cluster[n] = c
                new_neighbors = self.search_neighbor(n)

                if len(new_neighbors) >= self.minpts:
                    for nn in new_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)
            i = i+1

    def fit(self, x, eps, minpts):
        self.dataset = x
        self.map = {}
        self.eps = eps
        self.minpts = minpts
        self.cluster =  [0 for i in range(len(self.dataset))]

        c = 0

        for i in range(len(self.dataset)):
            if (self.cluster[i] == 0):
                neighbors = self.search_neighbor(i)

                if (len(neighbors) < self.minpts):
                    self.cluster[i] = -1
                else :
                    c = c +1
                    self.cluster[i] = c
                    self.create_cluster(neighbors, c)
        
        # self.create_mapping(c)
        # print(self.map)

    def predict(self, test_data):
        result = []

        for td in test_data:
            closest_data = 0
            min_distance = 9999
            for i in range(len(self.dataset)):
                distance = eucledian_distance(self.dataset[i],td)
                if (distance < min_distance and self.cluster[i]!= -1):
                    min_distance = distance
                    closest_data = i

            result.append(self.cluster[closest_data])
        return result

def create_dbscan_mapper(target, cluster):
    cluster_set = set(cluster)

    map = {}
    for c in cluster_set:
        counter = {}
        for j in range(len(target)):
            if cluster[j] == c:
                value = target[j]
                if value in counter.keys():
                    counter[value] = counter[value] + 1
                else :
                    counter[value] = 1
        max_key = max(counter.items(), key=operator.itemgetter(1))[0]
        map[c] = max_key
    return map

def count_accuracy(predict, y_test):
    correct_count = 0
    for i in range(len(predict)):
        if predict[i] == y_test[i]:
            correct_count = correct_count + 1
    
    return correct_count/len(predict)

if __name__ == "__main__" :
    print('tes')
    iris = datasets.load_iris()
    # print(iris.DESCR)
    print(iris.data)
    print(iris.target)
    print(iris.target_names)

    mydbscan = MyDBSCAN()
    scikitdbscan = DBSCAN(eps=0.41, min_samples=3)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
    mydbscan.fit(X_train, 0.41, 3)
    scikitdbscan.fit(X_train)

    mydbscan_map = create_dbscan_mapper(y_train,mydbscan.cluster)
    scikitdbscan_map = create_dbscan_mapper(y_train,scikitdbscan.labels_)

    mydbscan_predict = mydbscan.predict(X_test)
    scikitdbscan_predict = scikitdbscan.fit_predict(X_test)

    for i in range(len(mydbscan_predict)):
        mydbscan_predict[i] = mydbscan_map[mydbscan_predict[i]]
        scikitdbscan_predict[i] = scikitdbscan_map[scikitdbscan_predict[i]]

    print("My DBSCAN accuracy", count_accuracy(mydbscan_predict,y_test), "%")
    print("Scikit DBSCAN accuracy", count_accuracy(scikitdbscan_predict,y_test), "%")
    
    # print(dbscan.cluster)
