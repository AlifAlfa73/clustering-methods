from math import sqrt

def eucledian_distance(a,b):
    total = 0
    for i in range(len(a)):
        total = total + (a[i]-b[i]) ** 2
    
    return sqrt(total)