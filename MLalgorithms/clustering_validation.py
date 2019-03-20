import pandas as pd 
import numpy as np
def Fmeasure(clusters,partitions):
    F= 0
    contigencyTable = pd.crosstab(clusters, partitions)
    for i in range(contigencyTable.shape[0]):
        idx=contigencyTable.idxmax(1)
        F += (2/contigencyTable.shape[0]) * contigencyTable[i][idx[i]]/(pd.DataFrame.sum(contigencyTable[i],axis=0)+pd.DataFrame.sum(contigencyTable[:][idx[i]]))
    return F


def ConditionalEntropy(clusters,partitions):
    contigencyTable = pd.crosstab(clusters, partitions)
    H = 0
    for i in range(contigencyTable.shape[0]):
        ni=pd.DataFrame.sum(contigencyTable[i],axis=0)
        H += -(1/ni*contigencyTable.shape[0]) *np.sum(contigencyTable[i]*np.log2(contigencyTable[i]/ni))
    return H