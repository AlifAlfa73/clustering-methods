from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from utils import eucledian_distance
import collections
import numpy as np
import operator
import sys

class MyAgglomerative:
    def __init__(self):
        self.dataset = None
        self.cluster = []
        self.cluster_set = []
        self.linkage = None
        self.distance_matrix = None

    def generate_distance_matrix(self):
        distance_array = []
        data = self.dataset
        n = len(data)
        for i in range(n):
            for j in range(n):
                distance_array.append(eucledian_distance(data[i], data[j]))
        distance_matrix = np.array(distance_array).reshape(n, n)
        np.fill_diagonal(distance_matrix, sys.maxsize)
        return(distance_matrix)
 
    def cluster_average(self, cluster):
        average = []
        data = self.dataset
        for i in range(len(data[cluster[0]])):
            dimension_array = []
            for j in range(len(cluster)):
                dimension_array.append(data[cluster[j]][i])
            avg_dimension = sum(dimension_array)/len(dimension_array)
            average.append(avg_dimension)
        return(average)

    def generate_distance(self, cluster_1, cluster_2):
        data = self.dataset
        distance_array = []
        linkage = self.linkage
        for i in range(len(cluster_1)):
            for j in range(len(cluster_2)):
                distance_array.append(eucledian_distance(data[cluster_1[i]], data[cluster_2[j]]))
        if (linkage=="single"):
            return(min(distance_array))
        elif (linkage=="complete"):
            return(max(distance_array))
        elif (linkage=="average_group"):
            return(eucledian_distance(self.cluster_average(cluster_1), self.cluster_average(cluster_2)))
        elif (linkage=="average"):
            return(sum(distance_array)/len(distance_array))
        
    def generate_distance_point(self, point, cluster):
        data = self.dataset
        distance_array = []
        linkage = self.linkage
        for j in range(len(cluster)):
            distance_array.append(eucledian_distance(point, data[cluster[j]]))
        if (linkage=="single"):
            return(min(distance_array))
        elif (linkage=="complete"):
            return(max(distance_array))
        elif (linkage=="average_group"):
            return(eucledian_distance(point, self.cluster_average(cluster)))
        elif (linkage=="average"):
            return(sum(distance_array)/len(distance_array))
    
    def search_min_distance(self):
        min_val = sys.maxsize
        n = len(self.distance_matrix)
        for i in range(n):
            for j in range(n):
                if(self.distance_matrix[i][j] < min_val):
                    min_val = self.distance_matrix[i][j]
                    row_index = i
                    col_index = j
        min_point = (row_index, col_index)
        return(min_point)
    
    def update_distance_matrix(self, row_index, col_index):
        self.distance_matrix = np.delete(self.distance_matrix, col_index, 0)
        self.distance_matrix = np.delete(self.distance_matrix, col_index, 1)
        for i in range(len(self.cluster[col_index])):
            self.cluster[row_index].append(self.cluster[col_index][i])
        self.cluster.pop(col_index)
        for i in range(len(self.cluster)):
            distance = self.generate_distance(self.cluster[row_index], self.cluster[i])
            self.distance_matrix[row_index][i] = distance
            self.distance_matrix[i][row_index] = distance
    
    def generate_cluster(self):
        new_cluster = []
        for i in range(len(self.dataset)):
            new_cluster.append(0)
        for i in range(len(self.cluster)):
            for j in (self.cluster[i]):
                new_cluster[j] = i
        self.cluster_set = self.cluster
        self.cluster = new_cluster
        
    def fit(self, data, n_clusters, linkage):
        self.dataset = data
        self.linkage = linkage
        for i in range(len(self.dataset)):
            self.cluster.append([i])
        self.distance_matrix = self.generate_distance_matrix()
        while (len(self.cluster) > n_clusters):
            min_distance = self.search_min_distance()
            self.update_distance_matrix(min_distance[0], min_distance[1])
            np.fill_diagonal(self.distance_matrix, sys.maxsize)
        self.generate_cluster()
    
    def predict(self, test_data):
        result = []
        for td in test_data:
            distance_array = []
            for j in range(len(self.cluster_set)):
                distance_array.append(self.generate_distance_point(td, self.cluster_set[j]))
            min_val = sys.maxsize
            for j in range(len(distance_array)):
                if (distance_array[j] < min_val):
                    min_val = distance_array[j]
                    min_idx = j
            result.append(min_idx)
        return(result)
            
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
    
    return correct_count*100/len(predict)

if __name__ == "__main__" :
    iris = datasets.load_iris()

    myagglomerative = MyAgglomerative()
    scikitagglomerative = AgglomerativeClustering(linkage='average', n_clusters=3)
    
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=23)
    myagglomerative.fit(X_train, 3, "average")
    scikitagglomerative.fit(X_train)
    
    print("Clustering Result")
    print(myagglomerative.cluster)
    print(scikitagglomerative.labels_)

    myagglomerative_predict = myagglomerative.predict(X_test)
    scikitagglomerative_predict = scikitagglomerative.fit_predict(X_test)
    
    print("Predicting Result Before Mapping")
    print(myagglomerative_predict)
    print(scikitagglomerative_predict)
    
    myagglomerative_map = create_mapper(y_test, myagglomerative_predict)
    scikitagglomerative_map = create_mapper(y_test, scikitagglomerative_predict)
    
    print("Mapping Result")
    print(myagglomerative_map)
    print(scikitagglomerative_map)
    
    for i in range(len(myagglomerative_predict)):
        myagglomerative_predict[i] = myagglomerative_map[myagglomerative_predict[i]]
        scikitagglomerative_predict[i] = scikitagglomerative_map[scikitagglomerative_predict[i]]

    print("Solution")
    print(y_test)

    print("Predicting Result After Mapping")
    print(myagglomerative_predict)
    print(scikitagglomerative_predict)

    print("Evaluation")
    print("My Agglomerative accuracy", count_accuracy(myagglomerative_predict,y_test), "%")
    print("Scikit Agglomerative accuracy", count_accuracy(scikitagglomerative_predict,y_test), "%")