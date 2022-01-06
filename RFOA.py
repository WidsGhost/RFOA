# RFOA -> red fox optimization algortihm
#More details:
#Połap, Dawid, and Marcin Woźniak. "Red fox optimization algorithm." Expert Systems with Applications 166 (2021): 114107.
#Połap, Dawid, and Marcin Woźniak. "A hybridization of distributed policy and heuristic augmentation for improving federated learning approach." Neural Networks 146 (2022): 130-140.
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math

def sort(table1, table2):
    for i in range(len(table1)):
        for j in range(len(table1)):
            if table1[i]<table1[j]:
                    table1[i], table1[j]=table1[j], table1[i]
                    table2[i], table2[j]=table2[j], table2[i]
    return table1, table2

def fitnessFunction(point):
    tmp=0
    for i in range(len(point)):
        tmp+=point[i]**2
    return tmp

def rfoa(populationSize, dimension, iterations,L,R):
    foxes=[]
    fitness=[]
    
    #generate an initial population
    for i in range(populationSize):
        fox=[]
        for j in range(dimension):
            fox.append(np.random.uniform(L,R))
        foxes.append(fox)    
        
    for i in range(len(foxes)):
        fitness.append(fitnessFunction(foxes[i]))
    fitness, foxes=sort(fitness,foxes)

    
    
    for T in range(iterations):       
        #reproduction and leaving the herd            
        FromIndex=populationSize-0.05*populationSize
        for index in range(int(FromIndex),populationSize):
            habitatCenter=[]
            for i in range(dimension):
                habitatCenter.append((foxes[0][i]+foxes[1][i])/2)
          #  habitatDiameter=distance.euclidean(foxes[0],foxes[1])
            kappa=np.random.uniform(0,1)
            if kappa>=0.45:
                for i in range(dimension):
                    foxes[index][i]=np.random.uniform(L,R)
            else:
                for i in range(dimension):
                    foxes[index][i]=kappa*habitatCenter[i]
        
        #global phase - food searching
        for i in range(len(foxes)):
            #distances.append(distance.euclidean(foxes[i],foxes[0]))
            alpha=np.random.uniform(0,distance.euclidean(foxes[i],foxes[0]))
            for j in range(dimension):
                value=1
                if foxes[0][j]-foxes[i][j]<0:
                        value=-1
                if foxes[i][j]+alpha*value<R and foxes[i][j]+alpha*value>L:  
                    foxes[i][j]+=alpha*value
                elif foxes[i][j]-alpha*value<R and foxes[i][j]-alpha*value>L:  
                    foxes[i][j]-=alpha*value
        #local phase - traversing through the local habitat
        a=np.random.uniform(0,0.2)
        for i in range(len(foxes)):
            if np.random.uniform(0,1)>0.75:
                phi=[]
                for i in range(dimension):
                    phi.append(np.random.uniform(0,2*3.14))
                r=np.random.uniform(0,1)

                if phi[0] != 0:
                    r=a*math.sin(phi[0])/phi[0]
                for j in range(dimension):
                    if j==0:
                        a= foxes[i][j]+a*r*math.cos(phi[0])
                        b= foxes[i][j]-a*r*math.cos(phi[0])
                        if a<R and a>L:  
                            foxes[i][j]=a
                        elif b<R and b>L:  
                            foxes[i][j]=b
                    else:
                        for k in range(j):
                            if k!=j:
                                a= foxes[i][j]+a*r*math.sin(phi[j])
                                b= foxes[i][j]-a*r*math.sin(phi[j])
                                if a<R and a>L:  
                                    foxes[i][j]=a
                                elif b<R and b>L:  
                                    foxes[i][j]=b
                            else:
                                a= foxes[i][j]+a*r*math.cos(phi[j])
                                b= foxes[i][j]-a*r*math.cos(phi[j])
                                if a<R and a>L:  
                                    foxes[i][j]=a
                                elif b<R and b>L:  
                                    foxes[i][j]=b
        fitness.clear()
        for i in range(len(foxes)):
            fitness.append(fitnessFunction(foxes[i]))
        fitness, foxes=sort(fitness,foxes)
    return foxes[0]

results=[]
fitnesses=[]
for i in range(10):
    results.append(rfoa(10, 1, 100,-3,3))
    fitnesses.append(fitnessFunction(results[i]))
    print("f(",results[i],") =",fitnesses[i])

print("avg fitness =",sum(fitnesses)/10)