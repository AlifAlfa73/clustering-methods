from sklearn import datasets
from utils import eucledian_distance
import collections
import operator
from sklearn.cluster import DBSCAN
import numpy as np

# max(stats.iteritems(), key=operator.itemgetter(1))[0]

#DBSCAN Reference : https://github.com/chrisjmccormick/dbscan/blob/master/dbscan.py

class DBSCAN:
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

    def create_mapping(self, c):
        for i in range(1,c+1):
            counter = {}
            for j in range(len(self.target)):
                if self.cluster[j] == i:
                    value = self.target[j]
                    if value in counter.keys():
                        counter[value] = counter[value] + 1
                    else :
                        counter[value] = 1
            max_key = max(counter.items(), key=operator.itemgetter(1))[0]
            self.map[i] = max_key



    def fit(self, x, y, eps, minpts):
        self.dataset = x
        self.target = y
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
        
        self.create_mapping(c)
        print(self.map)

    def predict(self, test_data):
        closest_data = 0
        min_distance = 9999
        for i in range(len(self.dataset)):
            distance = eucledian_distance(self.dataset[i],test_data)
            if (distance < min_distance and self.cluster[i]!= -1):
                min_distance = distance
                closest_data = i

        print(self.cluster[closest_data])
        return self.map[self.cluster[closest_data]]


if __name__ == "__main__" :
    print('tes')
    iris = datasets.load_iris()
    # print(iris.DESCR)
    print(iris.data)
    print(iris.target)
    print(iris.target_names)

    dbscan = DBSCAN()
    dbscan.fit(iris.data, iris.target, 0.41, 3)
    print(dbscan.predict([5,3,5,2]))
    # print(dbscan.cluster)
