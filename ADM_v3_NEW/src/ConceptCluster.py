import numpy as np
import random
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.sparse import csr_matrix
import time
from WordBag import *
from ConceptRep import *


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

def KMPP(wl,cl,matrix,k):
	X=csr_matrix(matrix)
	true_k=min(k,len(cl))#maximum of 4 concept clusters per document
	km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
	t0 = time.time()
	km.fit(X)
	#print("done in %0.3fs" % (time.time() - t0))
	order_centroids = km.cluster_centers_.argsort()[:, ::-1]
	print ("Cluster Centres: ",km.cluster_centers_)
	#print(cl)#concept list
	#print("Cluster Results: ",km.labels_)
	C={}
	for i in range(true_k):
		C[i]=[]
	for i,c in enumerate(cl):
		C[km.labels_[i]].append(i)
	return C


def getSense(ChunkArr,start):
	# L=[[0., 0.5,0.2,0.5,0.2],
	#  [0.5, 0.0 ,0.1,0.4,0.2],
	#  [0.2, 0.1, 0.0 , 0.3 ,0.4],
	#  [0.5 ,0.4 ,0.3 ,0.0  ,0.2],
	#  [0.2, 0.2, 0.4 ,0.2, 0.0 ]]
	ChunkSense = []
	WordList = []
	print("Starting Sense computation at %s seconds"%(time.time()-start))
	for documentRep in ChunkArr:
		wl,cl,matrix= documentRep #k means ++
		#wl,cl,matrix,dv= documentRep # kmedoid
		#print("m is :",matrix,"\n")
		C = KMPP(wl,cl,matrix,4)
		# M,C = kMedoids(dv)
		#print ("\n\nPrinting C:\n\n",C)
		# print ('Mediods',M,'\n',"Clusters",C)
		C = {a:C[a] for a in sorted(C, key=lambda k: len(C[k]),reverse=True)[0:2]} # Get the dominant concept clusters
		#print ("\n\nPrinting C:\n\n",C)
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
	#print("ChunkSense : \n\n\n",ChunkSense)
	#print("WordList : \n\n\n",WordList)
	return (ChunkSense,WordList)

if __name__=="__main__":
	a = time.time()
	B = getBagOfWords("test.txt",start)
	ChunkArr = RepresentChunk(B,start)
	cs,wl = getSense(ChunkArr,start)
	sense,wl = getSense("test.txt",a)
	print(wl)
	b = time.time()
	print("It took %s seconds."%(b-a))