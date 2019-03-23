import pandas as pd 
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
def Fmeasure(clusters,partitions):
    F= 0
    contigencyTable =contingency_matrix(clusters,partitions)
    idx=contigencyTable.max(axis=1)
    for i in range(contigencyTable.shape[0]):
        nij = contigencyTable.max(axis=1)
        ni = np.sum(contigencyTable[i])
        ji = contigencyTable[:,idx[i]]
        mji = np.sum(ji)
        F += 2 * nij/(ni+mji)
    return F/contigencyTable.shape[0]

def ConditionalEntropy(clusters,partitions):
    contigencyTable = contingency_matrix(clusters, partitions)
    H = []
    Hci = 0
    for i in range(contigencyTable.shape[0]):
        ni = np.sum(contigencyTable[i])
        for j in range(contigencyTable.shape[1]):
            Hci -= contigencyTable[i][j]/ni * np.log10(contigencyTable[i][j]/ni)
        H.append(Hci)
    return H