from ClusterHelper import *
from ConceptCluster import *
from ConceptRep import *
from WordBag import *
import time
import _pickle as cPickle
import os.path
import string 
from collections import Counter
import multiprocessing

def getChunkCluster(F,start):
	B = getBagOfWords(F,start)
	ChunkArr = RepresentChunk(B,start)
	cs,wl = getSense(ChunkArr,start)
	print("\t\tInitilaizing Document Cluster setup...")
	wl = sorted(wl)
	wlen = len(wl)
	expvecl = []
	for i in cs:
		V,el = getConceptVecEntityList(i)
		expvec = getExpandedVector(V,el,wl)
		expvecl.append([v for k,v in expvec])
	print("\t\tExpanded the vector space.\n\t\tRunning 100000 iterations to find optimal cluster centers.")
	km = K_Means(expvecl,wlen,5)
	print("\t\t Cluster centers were found through PeiPei Li K_Means algorithm.")
	#print("indices\n\n",km)
	clusterofChunk = {}
	for i,x in enumerate(km):
		cluster = []
		for y in x:
			cluster.append(cs[y])
		clusterofChunk.update({i:cluster})
	print("\t\tCluster centers were formed successfully. Over to the next document.")
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

def TrainEnsembleMulti(arguments):
	TrainEnsemble(*arguments)

def TrainChunk(arguments):
	print(arguments)
	F,start = arguments
	costs = []
	filename = F
	if not (os.path.isfile("ChunkClusters.pkl")):
		print("Commencing Chunk training...")
		t0=time.time()
		x = filename
		print("Training for %s"%(x))
		if not (os.path.isfile(x+".pkl")):
			revel= getChunkCluster(x,start)
			print("\tDumping the partial pickle onto the disk...")
			cPickle.dump(revel,open(x+".pkl","wb"))
			print("\tDumped successfully.")
		print("Training for %s complete."%(x))
		t1=time.time()
		print("Time taken to Train Chunk: %s seconds"%(t1-t0))

def TrainEnsembleSeq(arguments):
	print(arguments)
	Flist,start = arguments
	costs = []
	filenames = Flist
	if not (os.path.isfile("ChunkClusters.pkl")):
		print("Commencing Ensemble training...")
		t0=time.time()
		ChunkClusters = []
		for x in filenames:
			print("Training for %s"%(x))
			if not (os.path.isfile(x+".pkl")):
				revel= getChunkCluster(x,start)
				ChunkClusters.append(revel)
				print("\tDumping the partial pickle onto the disk...")
				cPickle.dump(revel,open(x+".pkl","wb"))
				print("\tDumped successfully.")
			else:
				ChunkClusters.append(cPickle.load(open(x+".pkl","rb")))
			print("Training for %s complete."%(x))
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


def ClassifyChunk(F,Flist,ChunkClusters,start):
	B = getBagOfWords(F,start)
	ChunkArr = RepresentChunk(B,start)
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


def getTrainedEnsemble():
	filenames = ['engineering-sample.chunk','culture-arts-entertainment-sample.chunk', 'education-science-sample.chunk', 'politics-society-sample.chunk', 'computers-sample.chunk', 'sports-sample.chunk', 'health-sample.chunk','business-sample.chunk']
	start = time.time()
	nb_cpus = 4
	pool = multiprocessing.Pool(processes=nb_cpus)
	pool.map(TrainChunk, [(a, start) for a in filenames])
	TE = TrainEnsembleSeq((filenames,start))
	return TE
	#ClassifyChunk("test.txt",filenames,TE,start)
	# ChunkClusters = 
	# st,wl = getSense("singletest.txt")
	# for x in ChunkClusters:
	# 	costs.append(shortTextToChunkClusterMatch(x,st[0]))
	# print("The distance index is as follows: ",costs)
	# print("The context of the document seems to be %s "%(filenames[costs.index(min(costs))].replace("training_","").replace(".txt","")))

if __name__=="__main__":
	getTrainedEnsemble()
	