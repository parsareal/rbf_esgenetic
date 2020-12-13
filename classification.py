import random
import array

import math as math
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt 
from sklearn import linear_model, datasets, preprocessing

numberOfClusters = 4
numberOfCenters = 7
n_features = 2
trainDataNum = 1200
testDataNum = 4000

#                               Read file

import csv
with open('4clstrain1200.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data = np.empty((trainDataNum, (n_features+1)))    
    X = np.empty((trainDataNum, n_features))
    y = np.empty(trainDataNum)    
    # print(X_train)
    i = -1
    for row in readCSV:
        if i != -1:
            a = float(row[0])
            b = float(row[1])
            c = float(row[2])
            c = c - 1
            abc = np.array((a, b, c))
            data[i] = abc
            # y_test[i] = c
        i = i + 1
    
    np.random.shuffle(data)
    X = data[:, 0:n_features]
    y = data[:, n_features]

with open('4clstest4000.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data = np.empty((testDataNum, (n_features+1)))    
    X_test = np.empty((testDataNum, n_features))
    y_test = np.empty(testDataNum)    
    # print(X_train)
    i = -1
    for row in readCSV:
        if i != -1:
            a = float(row[0])
            b = float(row[1])
            c = float(row[2])
            c = c - 1
            abc = np.array((a, b, c))
            data[i] = abc
            # y_test[i] = c
        i = i + 1
    
    np.random.shuffle(data)
    X_test = data[:, 0:n_features]
    y_test = data[:, n_features]

# print(X)
X = preprocessing.scale(X)
X_test = preprocessing.scale(X_test)

# print(X)
# print(y)

y_star = np.empty((trainDataNum, numberOfClusters))
for i in range(numberOfClusters):
    for j in range(y.size):
        if y[j] == i:
            y_star[j, i] = 1
        else:
            y_star[j, i] = 0

y_star_test = np.empty((testDataNum, numberOfClusters))
for i in range(numberOfClusters):
    for j in range(y_test.size):
        if y_test[j] == i:
            y_star_test[j, i] = 1
        else:
            y_star_test[j, i] = 0





def evaluateInd(individual):
    v = np.empty((numberOfCenters,n_features))
    gamas = np.empty(numberOfCenters)
    for i in range(numberOfCenters):
            v[i] = individual[n_features*i:(n_features*i+n_features)]
            gamas[i] = individual[n_features*numberOfClusters + i]
          
    G = np.empty((trainDataNum, numberOfCenters))
    # print(gamas)
    for i in range(X.shape[0]):
        data = X[i]
        for j in range(v.shape[0]):
            dist = np.linalg.norm(data-v[j]) ** 2
            # dist = dist / 15
            dist = dist * gamas[j]
            # print(dist)
            G[i, j] = math.exp(-dist)

    # print(G.transpose())
    # print(G)
    g = G.transpose().dot(G) + np.eye(numberOfCenters)
    invg = np.linalg.inv(g)
    weights = (invg.dot(G.transpose())).dot(y_star)

    ylabel = G.dot(weights)
   
    x = np.sum(np.sign(np.abs(np.argmax(y_star, axis=1) - np.argmax(ylabel, axis=1))))
    err = float(x)/len(ylabel)
    
    return err,



 
def printRes(individual):
    v = np.empty((numberOfCenters,n_features))
    gamas = np.empty(numberOfCenters)

    
    for i in range(numberOfCenters):
            v[i] = individual[n_features*i:(n_features*i+n_features)]
            gamas[i] = individual[n_features*numberOfCenters + i]
        
    G = np.empty((testDataNum, numberOfCenters))

    for i in range(X_test.shape[0]):
        data = X_test[i]
        for j in range(v.shape[0]):
            # dist = np.linalg.norm(data-v[j]) ** 2
            dist = np.dot(data - v[j], data - v[j])
            dist = dist * gamas[j]
            # print(dist)
            G[i, j] = math.exp(-dist)


    g = G.transpose().dot(G)
    invg = np.linalg.inv(g)
    weights = (invg.dot(G.transpose())).dot(y_star_test)

    ylabel = G.dot(weights)

    # print(ylabel)
    # print(y_star_test)
    yLabel = np.argmax(ylabel, axis=1)
    y_star = np.argmax(y_star_test, axis=1)

    plotTrueFalse(X_test, yLabel, y_star, v)
    # plotCusters(X_test, yLabel, y_star, v)


def plotTrueFalse(X, yLabel, y_star, v):
    eqs = float(np.sum(yLabel == y_star))
    ys = float(len(yLabel))
    plt.title('Test Acc = ' + str(eqs/ys))
    colors = ['r', 'g']
    for i in range(len(yLabel)):
        if (y_star[i] == yLabel[i]):
            plt.scatter(X[i, 0], X[i, 1], c=colors[1])   
        else:
            plt.scatter(X[i, 0], X[i, 1], c=colors[0])   

    plt.plot(v[:, 0], v[:, 1], 'y+', markersize=14)
    plt.show()


def plotCusters(X, yLabel, y_star, v):
    eqs = float(np.sum(yLabel == y_star))
    ys = float(len(yLabel))
    plt.title('Test Acc = ' + str(eqs/ys))
    colors = ['r', 'g', 'y', 'black']
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], c=colors[yLabel[i]])  
    plt.plot(v[:, 0], v[:, 1], '+')

    plt.show()








creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d",
               fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

def initES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

IND_SIZE = numberOfCenters * (n_features + 1)
# MIN_VALUE, MAX_VALUE = -2., 2.
# MIN_STRAT, MAX_STRAT = -1., 1. 
MIN_VALUE = np.min(X)
MAX_VALUE = np.max(X)
MIN_STRAT = -1
MAX_STRAT = 1

def crossover(ind1, ind2):
        res1, res2 = tools.cxESBlend(ind1, ind2, alpha=0.1)
        return res1, res2

def checkStrategy(minstrategy):
        def decorator(func):
            def wrappper(*args, **kargs):
                children = func(*args, **kargs)
                for child in children:
                    for i, s in enumerate(child.strategy):
                        if s < minstrategy:
                            child.strategy[i] = minstrategy
                return children

            return wrappper

        return decorator

toolbox = base.Toolbox()
toolbox.register("individual", initES, creator.Individual,
                 creator.Strategy, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRAT, 
                 MAX_STRAT)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluateInd)
toolbox.register("mate", crossover)

toolbox.decorate("mate", checkStrategy(MIN_STRAT))
toolbox.decorate("mutate", checkStrategy(MAX_STRAT))

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
hof = tools.HallOfFame(1)

pop = toolbox.population(n=10)

# bestGen = algorithms.eaMuCommaLambda(pop, toolbox, mu=10, lambda_= 100,
#                                                   cxpb=0.6, mutpb=0.3, ngen=10, stats=stats, halloffame=hof)
# print(hof[0])

ind = [0.28174991564539303, 0.21996855184314806, -0.71543996762018, 0.14829802524847588, -0.5732294309681212, -0.21143047468863854, -0.3891915206667921, 1.561516212448428, -0.21333653611785403, 1.3637220137127029, 1.1253858001678292, -0.11595603143269623, 0.22154631908781536, 0.1394004740686266, 0.5825458990155138, 0.8410803181579813, -0.723863638805625, 1.4197299512868966, 0.6073594868385371, 1.9183797635598436, 0.7593072122213335]
err = evaluateInd(ind)
# print(ind)
print(err)
printRes(ind)

