###Implémentation

#Chargement de dépendances
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd
import copy

#Discrétisation
A=0
B=500
N=101 #Nombre de points de discrétisayion
Delta = (B-A)/(N-1)
discretization_indexes=np.arange(N)
discretization=discretization_indexes*Delta

#Paramètres du modèle

mu=-5
a=50
sigma2=12

#Données

observation_indexes= [0,20,40,60,80,100]
depth=[0,-4,-12.8,-1,-6.5,0]

unknown_indexes=list(set(discretization_indexes)-set(observation_indexes))

###question 1###
def com(distance, a, coef):
        return coef*np.exp(distance/a)
###question 2###

Mat_distance=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        Mat_distance[i][j]=np.abs(discretization[i]-discretization[j])
#print(Mat_distance)
###question 3###
Cov_Z=com(Mat_distance,a,sigma2)

###question 4###
k=len(observation_indexes)


Cov_obs=np.zeros((k,k))
Cov_unknow_obs=np.zeros((N-k,k))
Cov_unknow=np.zeros((N-k,N-k))
for i in range(k):
    for j in range(k):
        Cov_obs[i][j]=Cov_Z[observation_indexes[i]][observation_indexes[j]]

for i in range(N-k):
    for j in range(N-k):
        Cov_unknow[i][j]=Cov_Z[unknown_indexes[i]][unknown_indexes[j]]

for i in range(N-k):
    for j in range(k):
        Cov_unknow_obs[i][j]=Cov_Z[unknown_indexes[i]][observation_indexes[j]]

###question 5###
m_z= np.full(N-k,mu)
m_y= np.full(k,mu)
Cov_obs_inv=np.linalg.inv(Cov_obs)
Q=np.zeros(k)
for i in range(k):
    Q[i]=depth[i]
print(Q)
B=Q-m_y
C=np.dot(Cov_unknow_obs,Cov_obs_inv)
A=np.dot(C,B)
esp_cond=m_z+A

Liste_cond=[]
for i in esp_cond:
    Liste_cond.append(i)

#plt.scatter(unknown_indexes,esp_cond,color='b')
#plt.scatter(observation_indexes,depth,color='r')
# plt.show()

### Question 6###
Cov_obs_unknow=np.transpose(Cov_unknow_obs)

Cov_cond = Cov_unknow-np.dot(C,Cov_obs_unknow)

# Y=[]
# for i in range(N-k):
#     Y.append(Cov_cond[i][i])
# plt.plot(unknown_indexes,Y)
# plt.show()

### question 7 ###

H=-np.around(Cov_cond,4)
L=-np.linalg.cholesky(H)

def simulation(n):
    R=[]
    for j in range(n):
        P=[]
        Z=[]
        for u in range(N-k):
            P.append(np.sqrt(-2*np.log(rd.random()))*np.cos(2*np.pi*rd.random()))
        T=  np.dot(L,P)
        Z=esp_cond+T
        R.append(Z)
    return R

# plt.scatter(,Z)
# plt.show()

### question 8 ###

def longeur (Liste_prof, pas):
    S=0
    for i in range(len(Liste_prof)-1):
        S+= np.sqrt(pas**2 + (Liste_prof[i+1]-Liste_prof[i])**2)
    return S

### question 9 ###

def profondeur(Liste):
    L=[]
    for i in Liste:
        L.append(i)
    for i in range(len(observation_indexes)):
        L.insert(observation_indexes[i],depth[i])
    return L



R=simulation(10000)
Longueur_esp_cond = longeur(profondeur(Liste_cond),5)
Moy_longueur = 0
print(profondeur(R[1]))
for i in R:
    Moy_longueur+=longeur(profondeur(i),5)
Moy_longueur=Moy_longueur/len(R)
# plt.scatter(discretization,profondeur(R[1]))
# plt.scatter(discretization,profondeur(Liste_cond))
# plt.show()
print(Longueur_esp_cond)
print(Moy_longueur)




