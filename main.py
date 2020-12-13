import random
import array
import pandas
import math as math
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt 
from sklearn import linear_model, datasets, preprocessing

numberOfCenters = 10
n_features = 3
numberOfData = 1500
trainDataNum = 900
testDataNum = 1500


import csv
with open('regdata1500.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data = np.empty((numberOfData, (n_features+1)))    
    # X_test = np.empty((2000, 3))
    # y_test = np.empty(2000)    
    # print(X_train)
    i = -1
    for row in readCSV:
        if i != -1:
            a = float(row[0])
            b = float(row[1])
            c = float(row[2])
            d = float(row[3])
            # c = c - 1
            abc = np.array((a, b, c, d))
            data[i] = abc
            # y_test[i] = c
        i = i + 1
    
    # min_max_scaler = preprocessing.MinMaxScaler()
    # np_scaled = min_max_scaler.fit_transform(data)
    # np.random.shuffle(np_scaled)
    # np.random.shuffle(data)
    
    X = data[:, 0:n_features]
    y = data[:, n_features]

    y = preprocessing.scale(y)
    X = preprocessing.scale(X)


X_train = X[0:trainDataNum]
y_train = y[0:trainDataNum]

X_test = X[0:numberOfData]
y_test = y[0:numberOfData]

# print(X)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d",
               fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

def initES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

IND_SIZE = numberOfCenters * (1 + n_features)
MIN_VALUE, MAX_VALUE = -1, 1.
MIN_STRAT, MAX_STRAT = 0.5, 3. 

toolbox = base.Toolbox()
toolbox.register("individual", initES, creator.Individual,
                 creator.Strategy, IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRAT, 
                 MAX_STRAT)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# def evaluateInd(individual):
#     # Do some computation
#     v = np.empty((numberOfCenters,2))
#     for i in range(numberOfCenters):
#             v[i] = individual[2*i:(2*i+2)]
# #     v[0] = individual[0:2]
# #     v[1] = individual[2:4]
#     G = np.empty((numberOfData, numberOfCenters))

#     for i in range(X.shape[0]):
#         data = X[i]
#         for j in range(v.shape[0]):
#             dist = np.linalg.norm(data-v[j]) ** 2
#             dist = dist / 10
#             G[i, j] = math.exp(-dist)

#     # print(G.transpose())
#     # print(G)
#     g = G.transpose().dot(G)
#     invg = np.linalg.inv(g)
#     weights = (invg.dot(G.transpose())).dot(y)

#     ylabel = G.dot(weights)

#     # print(ylabel)
#     sub = np.subtract(ylabel, y_pred)
#     err = sub.transpose().dot(sub)
#     err = err / 2
#     # print(err)

#     # result = individual[0]
#     return err,


def evaluateInd(individual):
    v = np.empty((numberOfCenters,n_features))
    gamas = np.empty(numberOfCenters)
    for i in range(numberOfCenters):
            v[i] = individual[n_features*i:(n_features*i+n_features)]
            gamas[i] = individual[n_features*numberOfCenters + i]
            
    G = np.empty((trainDataNum, numberOfCenters))
    # print(gamas)
    for i in range(X_train.shape[0]):
        data = X_train[i]
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
    weights = (invg.dot(G.transpose())).dot(y_train)

    ylabel = G.dot(weights)

    # print(ylabel)
    sub = np.subtract(ylabel, y_train)
    err = sub.transpose().dot(sub)
    err = err / 2
    # print(err)

    # result = individual[0]
    return err,


# individual = [15, 5, 10, 20]
# print(evaluateInd(individual))

def plotRes(individual):
    v = np.empty((numberOfCenters,n_features))
    gamas = np.empty(numberOfCenters)

    
    for i in range(numberOfCenters):
            v[i] = individual[n_features*i:(n_features*i+n_features)]
            gamas[i] = individual[n_features*numberOfCenters + i]
            # if gamas[i] > 0.5:
            #     gamas[i] = 0.5
            # if gamas[i] < -10:
            #     gamas[i] = 0    

    G = np.empty((testDataNum, numberOfCenters))

    for i in range(X_test.shape[0]):
        data = X_test[i]
        for j in range(v.shape[0]):
            # dist = np.linalg.norm(data-v[j]) ** 2
            dist = np.dot(data - v[j], data - v[j])
            dist = dist * gamas[j]
            # print(dist)
            G[i, j] = math.exp(-dist)

    # print(G.transpose())
    # print(G)
    g = G.transpose().dot(G)
    invg = np.linalg.inv(g)
    weights = (invg.dot(G.transpose())).dot(y_test)

    ylabel = G.dot(weights)
    

    plt.plot(y_test, "b")
    plt.plot(ylabel, "r")
    plt.show()
#     # print(ylabel)
#     sub = np.subtract(ylabel, y)
#     err = sub.transpose().dot(sub)
#     err = err / 2
#     # print(err)

    # result = individual[0]
    


# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("evaluate", evaluateInd)

# stats = tools.Statistics(key=lambda ind: ind.fitness.values)
# stats.register("avg", np.mean)
# stats.register("std", np.std)
# stats.register("min", np.min)
# stats.register("max", np.max)
# hof = tools.HallOfFame(1)

# pop = toolbox.population(n=20)
# bestGen = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof)

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
#                                                   cxpb=0.6, mutpb=0.3, ngen=20, stats=stats, halloffame=hof)

ind = [-0.3636183229399603, -0.32567531278247214, -0.47086735076799335, 0.09669417616240242, -0.3736057020390817, 0.2088897796876335, 0.2367070630664668, -0.6700734101255411, 0.278862805300256, 0.1678025202019564, -0.03985796171822853, -0.694789315023843, -0.22791266508807845, 0.8812567954430561, 0.5518095146082201, 0.36379419241576283, 0.3558926722345013, 0.7512460227301743, 0.5529821388945895, 0.4605353381670096, -0.6372826064852462, 0.10539465319639085, -0.20288249413921425, 0.05956085170463574, 0.4935221557313242, 0.6033951902921257, -0.7041949362631017, -0.07487524287285793, -0.22984430847850584, -0.6546856696022036, -0.13012529042189352, 0.5383327384907519, -0.10863218639971738, 0.3253582849198116, -0.10600147016618416, 0.17838910326633525, -0.19032680342600772, -0.1602735171870961, -0.11522344020138489, -0.23715722460271654]
# err = evaluateInd(hof[0])
err = evaluateInd(ind)
plotRes(ind)
# print(hof[0])
print(err)
# print(y)

