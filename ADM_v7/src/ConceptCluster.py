import numpy as np
import random
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
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
	C={}
	X=csr_matrix(matrix)
	if len(X.shape)==0 or X.shape[0]==0:
		return C
	true_k=min(k,len(cl))#maximum of 4 concept clusters per document
	km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
	t0 = time.time()
	try:
		km.fit(X)

		# h = .02  
		# reduced_data = PCA(n_components=2).fit_transform(X.toarray())
		# kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
		# kmeans.fit(reduced_data)

		# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
		# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
		# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		# Z= kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

		# Z = Z.reshape(xx.shape)
		# plt.figure(1)
		# plt.clf()

		# plt.imshow(Z, interpolation='nearest',
		#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
		#            cmap=plt.cm.Paired,
		#            aspect='auto', origin='lower')

		# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
		# # Plot the centroids as a white X
		# centroids = kmeans.cluster_centers_
		# plt.scatter(centroids[:, 0], centroids[:, 1],
		#             marker='x', s=169, linewidths=3,
		#             color='w', zorder=10)
		# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
		#           'Centroids are marked with white cross')
		# plt.xlim(x_min, x_max)
		# plt.ylim(y_min, y_max)
		# plt.xticks(())
		# plt.yticks(())
		# plt.show()

		#print("done in %0.3fs" % (time.time() - t0))
		order_centroids = km.cluster_centers_.argsort()[:, ::-1]
		#print ("Cluster Centres: ",km.cluster_centers_)
		#print(cl)#concept list
		#print("Cluster Results: ",km.labels_)
		for i in range(true_k):
			C[i]=[]
		for i,c in enumerate(cl):
			C[km.labels_[i]].append(i)
		# print (C)
	except ValueError as e:
		return {}
	finally:
		print(time.ctime())
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
		if not matrix:
			C = {}
		else:
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
	ChunkArr = RepresentChunk(B,start,"test.txt")
	cs,wl = getSense(ChunkArr,start)
	sense,wl = getSense("test.txt",a)
	print(wl)
	b = time.time()
	print("It took %s seconds."%(b-a))