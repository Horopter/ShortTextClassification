import numpy as np
import random
from ConceptSet import getConceptSet
from sklearn import metrics


def kMedoids(D, k=6, maxItr=100):
	DM=np.array(D)
	n = DM.shape[0]
	if k > n:
		raise Exception('K>N')
	random.seed(123)
	M = np.array(list(range(n)))
	np.random.shuffle(M)
	M = np.sort(M[:k])
	Mnew = np.copy(M)

	C = {}
	for t in range(maxItr):
		minInd = np.argmin(DM[:,M], axis=1)
		for i in range(k):
			minInd[M[i]] = i
		for clust_i in range(k):
			C[clust_i] = np.where(minInd==clust_i)[0]
		for clust_i in range(k):
			minInd = np.mean(DM[np.ix_(C[clust_i],C[clust_i])],axis=1)
			j = np.argmin(minInd)
			Mnew[clust_i] = C[clust_i][j]
		np.sort(Mnew)
		if np.array_equal(M, Mnew):
			break
		M = np.copy(Mnew)
	else:
		minInd = np.argmin(DM[:,M], axis=1)
		for clust_i in range(k):
			C[clust_i] = np.where(minInd==clust_i)[0]

	return M, C

def getSense(filename):
	# L=[[0., 0.5,0.2,0.5,0.2],
	#  [0.5, 0.0 ,0.1,0.4,0.2],
	#  [0.2, 0.1, 0.0 , 0.3 ,0.4],
	#  [0.5 ,0.4 ,0.3 ,0.0  ,0.2],
	#  [0.2, 0.2, 0.4 ,0.2, 0.0 ]]
	ListofConcepts,DistanceMatrix,FeatureVector = getConceptSet(filename)
	M,C = kMedoids(DistanceMatrix)
	Cn = {}
	for k,v in C.items():
		m = []
		for l in v:
			m.append(ListofConcepts[l])
		Cn.update({k:m})
	GM = [0 for x in range(len(M))]
	for i in range(len(M)):
			GM[i] = FeatureVector[i]
	return (ListofConcepts,DistanceMatrix,FeatureVector,M,C,Cn,GM)

if __name__=="__main__":
	sense = getSense("test.txt")