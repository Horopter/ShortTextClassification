from ClusterHelper import *
from ConceptCluster import *

def getChunkCluster(F):
	cs,wl = getSense(F)
	wl = sorted(wl)
	wlen = len(wl)
	expvecl = []
	for i in cs:
		V,el = getConceptVecEntityList(i)
		expvec = getExpandedVector(V,el,wl)
		expvecl.append([v for k,v in expvec])
	km = KMeans(expvecl,wlen)
	#print("indices\n\n",km)
	clusterofChunk = {}
	for i,x in enumerate(km):
		cluster = []
		for y in x:
			cluster.append(cs[y])
		clusterofChunk.update({i:cluster})
	#print("\n\n\nclusterofChunk\n\n",clusterofChunk)
	return clusterofChunk

def shortTextToChunkClusterMatch(coc,st):
	tcost = 1000000000000
	epsilon = (float(0.9999)/tcost)
	ind = -1
	for j,docClu in coc.items():
		cost=0
		if j==0:
			print("DocClu Schema :",docClu,"\n\n\n\n\n")
		for i,doc in enumerate(docClu):
			cost+= SemDistShortText(doc,st)
		cost = cost / (len(docClu)+epsilon)
		if cost < tcost:
			tcost = cost
			ind = j
	return tcost

if __name __ == "__main__":
	filenames = []
	st = None
	ChunkClusters = []
	costs = []
	for x in filenames:
		ChunkClusters.append(getChunkCluster(x))
	for x in ChunkClusters:
		costs.append(shortTextToChunkClusterMatch(x,st))