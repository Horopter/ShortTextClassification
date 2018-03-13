from ClusterHelper import *
from ConceptCluster import *
from AnyConceptRep import *
from ConceptRep import *
import time
import _pickle as cPickle
import os.path
import string 
from collections import Counter
from AnyWordBag import getAnyBagOfWords
from WordBag import *

def getChunkCluster(F,start):
	B = getBagOfWords(F,start)
	ChunkArr = RepresentChunk(B,start)
	cs,wl = getSense(ChunkArr,start)
	wl = sorted(wl)
	wlen = len(wl)
	expvecl = []
	for i in cs:
		V,el = getConceptVecEntityList(i)
		expvec = getExpandedVector(V,el,wl)
		expvecl.append([v for k,v in expvec])
	km = K_Means(expvecl,wlen,5)
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
			pass
			#print("DocClu Schema :",docClu,"\n\n\n\n\n")
		for i,doc in enumerate(docClu):
			sem = (SemDistShortText(doc,st))
			#print(sem)
			cost+= sem
		cost = cost / (len(docClu)+epsilon)
		#print("cost: ",cost," len(docClu) ",len(docClu),"\n\n\n")
		if cost < tcost and cost != 0:
			tcost = cost
			ind = j
	#print("Document Cluster Index = ",ind,"\n\n\n")
	return tcost

def TrainEnsemble(Flist,start):
	costs = []
	filenames = Flist
	if not (os.path.isfile("ChunkClusters.pkl")):
		print("Commencing Ensemble training...")
		t0=time.time()
		ChunkClusters = []
		for x in filenames:
			ChunkClusters.append(getChunkCluster(x,start))
		fp = open("ChunkClusters.pkl",'wb')
		cPickle.dump(ChunkClusters,fp)
		t1=time.time()
		print("Time taken to Train Ensemble: %s seconds"%(t1-t0))
	if os.path.isfile("ChunkClusters.pkl"):
		print("Found trained ensemble...")
		fp = open("ChunkClusters.pkl",'rb')
		ChunkClusters = cPickle.load(fp)
		print("Loaded the trained ensemble...")
	return ChunkClusters


def ClassifyChunk(F,Flist,ChunkClusters,TP,start):
	B = getAnyBagOfWords(F,start,TP)
	ChunkArr = AnyRepresentChunk(B,start,4*TP/5)#1/5 time taken by Bag of words 
	stl,wl = getSense(ChunkArr,start)
	annotation = []
	for st in stl:
		costs= []
		for x in ChunkClusters:
			costs.append(shortTextToChunkClusterMatch(x,st))
		annotation.append(Flist[costs.index(min(costs))].replace(".chunk",""))
	most_common,num_most_common = Counter(annotation).most_common(1)[0]
	print("The context of the chunk seems to be %s "%most_common)
	#print(max(groupby(sorted(annotation)), key=lambda (v):len(list(v),-annotation.index(x)))[0])


if __name__ == "__main__":
	filenames = ['culture-arts-entertainment-sample.chunk', 'business-sample.chunk', 'politics-society-sample.chunk', 'engineering-sample.chunk', 'computers-sample.chunk', 'sports-sample.chunk', 'health-sample.chunk', 'education-science-sample.chunk']
	start = time.time()
	TE = TrainEnsemble(filenames,start)
	#timeToProcess=random.randint(200,400)
	timeToProcess=190
	print("GIVEN TIME TO PROCESS: ",timeToProcess," secs")
	ClassifyChunk("test.txt",filenames,TE,timeToProcess,start)
	# ChunkClusters = 
	# st,wl = getSense("singletest.txt")
	# for x in ChunkClusters:
	# 	costs.append(shortTextToChunkClusterMatch(x,st[0]))
	# print("The distance index is as follows: ",costs)
	# print("The context of the document seems to be %s "%(filenames[costs.index(min(costs))].replace("training_","").replace(".txt","")))
	