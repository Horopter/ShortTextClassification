import numpy as np
import random
from ConceptRep import RepresentChunk
from sklearn import metrics
import time


def kMedoids(D, k=3, maxItr=100): #In a short-text number of clusters will be incredibly less.
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
	start = time.time()
	ChunkArr = RepresentChunk(filename)
	ChunkSense = []
	WordList = []
	print("Starting k-medoid computation at %s seconds"%(time.time()-start))
	for documentRep in ChunkArr:
		wl,cl,matrix,dv = documentRep
		M,C = kMedoids(dv)
		Cn = {}
		GM=[]
		for k,v in C.items():
			md = []
			y=[]
			for l in v:
				md.append(cl[l])
				luke = {}
				for i,sh in enumerate(matrix[l]):
					luke.update({wl[i]:sh})
				y.append((cl[l],luke))
			Cn.update({k:md})
			GM.append((k,y))
		#print(GM)
		ChunkSense.append(GM)
		WordList.extend(wl)
	end = time.time()
	print("Sense Representation Complete at %s seconds"%(end-start))
	return (ChunkSense,WordList)

if __name__=="__main__":
	a = time.time()
	sense,wl = getSense("test.txt")
	print(wl)
	b = time.time()
	print("It took %s seconds."%(b-a))