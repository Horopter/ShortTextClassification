"""
Concept Cluster per a document example
[
	(
		0, 
		[
			('business', {'multi': 234.25893061023473, 'products': 47.466996391615886, 'precision': 308.68441111734654, 'clients': 1000000, 'manufacturing': 64.10708488183323, 'spindle': 395.467134880532}), 
			('production', {'multi': 233.81061234291516, 'products': 44.40403158671639, 'precision': 265.494882485549, 'clients': 1000000, 'manufacturing': 64.0312499318277, 'spindle': 298.65654987488546})
		]
	), 
	(
		1, 
		[
			('area', {'multi': 240.22903542814225, 'products': 60.47552500793344, 'precision': 322.6831869003666, 'clients': 439.17818762303637, 'manufacturing': 79.31013620848942, 'spindle': 1000000}), 
			('company', {'multi': 239.89774211300664, 'products': 41.45720816019658, 'precision': 301.5214454033667, 'clients': 374.4432173526751, 'manufacturing': 80.3127393584483, 'spindle': 439.2891548846454}), 
			('factor', {'multi': 237.41097699225918, 'products': 56.94487710500109, 'precision': 324.3250664416908, 'clients': 361.29985804441685, 'manufacturing': 89.08487321266327, 'spindle': 355.72095890853245}), 
			('information', {'multi': 258.2158915481757, 'products': 55.08651337837963, 'precision': 332.32436675703485, 'clients': 357.41019972903325, 'manufacturing': 95.87915817350276, 'spindle': 382.49603270103177})
		]
	), 
	(
		2, 
		[
			('item', {'multi': 266.0011913934977, 'products': 55.0947625007118, 'precision': 358.0255219460524, 'clients': 413.7760870259337, 'manufacturing': 109.75201681101652, 'spindle': 330.4427536926003}), 
			('machine', {'multi': 204.55801393431224, 'products': 54.399375291088184, 'precision': 212.41384033779957, 'clients': 338.95666866095684, 'manufacturing': 86.47881984700439, 'spindle': 206.36795140874858}), 
			('product', {'multi': 227.33459036833904, 'products': 44.13127495622749, 'precision': 273.65784081694426, 'clients': 394.8949124843201, 'manufacturing': 89.35964722701996, 'spindle': 301.37692262825124}), 
			('services', {'multi': 1000000, 'products': 1000000, 'precision': 1000000, 'clients': 1000000, 'manufacturing': 1000000, 'spindle': 1000000})
		]
	)
]

"""
from collections import OrderedDict
import math
import random
import copy
import scipy
from WordBag import *
from ConceptRep import *
from ConceptCluster import *
import numpy

def getConceptVecEntityList(shortText):#shortText is one document
	ConceptVec = []
	klen=0
	for tuplet in shortText:
		arr = {}
		conceptClusterIndex,conceptCluster=tuplet
		klen=len(conceptCluster)
		for conceptTuple in conceptCluster:
			conceptName,entityDict = conceptTuple
			keys = entityDict.keys()
			for entity in keys:
				if entity in arr:
					arr.update({entity:arr[entity]+entityDict[entity]})
				else:
					arr.update({entity:entityDict[entity]})
		for ent,val in arr.items():
			arr.update({ent:val/klen})
		ConceptVec.append(arr)
	entityList = []
	ConceptVec = [x for x in ConceptVec if x] # remove empty clusters
	if len(ConceptVec) > 0:
		entityList = ConceptVec[0].keys()
	#print(ConceptVec)
	return (ConceptVec,entityList)

def DotProduct(v1, v2):
	total = 0
	for a in v1:
		for b in v2:
			total += (a*b)
	return total

def magnitude(rvec):
	total = 0
	for v1 in rvec:
		for v2 in rvec:
			total += DotProduct(v1,v2)
	return math.sqrt(total)

def magnitude2(rvec):
	total = 0
	total = np.dot(rvec,rvec).ravel().sum()
	return math.sqrt(total)

def getDocumentClusterCenter(docList):
	ConceptVecEntityListArr = []
	for shortText in docList:
		ConceptVecEntityListArr.append(getConceptVecEntityList(shortText))
	wl = [x[1] for x in ConceptVecEntityListArr]
	wl = [item for sublist in wl for item in sublist]
	ExpVecL = []
	for y in ConceptVecEntityListArr:
		cv,el = y
		ExpVecL.append(getExpandedVector(cv,el,wl))
	listdict = []
	for x in ExpVecL:
		listdict.append(dict(x))
	centroid = {}
	for entity in wl:
		for d in listdict:
			if entity in centroid.keys():
				centroid[entity] += d[entity]
			else:
				centroid.update({entity : 0})
	for key,val in centroid.items():
		centroid[key]= val/max(len(ConceptVecEntityListArr),1)
	return centroid

def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def getExpandedVector(V,el,wl):
	d = {}
	for v in V:
		for e in el:
			if e in d.keys() and e in v.keys():
				d.update({e:d[e]+v[e]})
			elif e in v.keys():
				d.update({e:v[e]})
			else:
				d.update({e:0})
	expvec = {}
	for e in wl:
		if e in expvec.keys() and e in d.keys():
			expvec.update({e:expvec[e]+d[e]})
		elif e in d.keys():
			expvec.update({e:d[e]})
		else:
			expvec.update({e:0})
	return sorted(expvec.items(), key=lambda x: x[0])

def ShortTextChunkSemDistance(chunk,st):
	ev1 = getDocumentClusterCenter(chunk)
	#ev1 = sorted(ev1.items(), key=lambda x: x[0])
	#print("ev1\t\t:",ev1)
	ev2 = getDocumentClusterCenter([st])
	#ev2 = sorted(ev2.items(), key=lambda x: x[0])
	#print("ev2:\t\t",ev2)
	items = ev2.keys()
	entities = ev1.keys()
	l = intersection(items,entities)
	if len(l)==0:
		return 0
	else:
		return len(l)


def addAll(indexList,vector):
	selectedVector = []
	for x in indexList:
		selectedVector.append(vector[x])
	return list(map(sum, zip(*selectedVector)))

def vecF(oc,nc):
	epsilon = 0.00001
	return np.multiply.outer(oc,nc).ravel().sum()/(magnitude2(oc)*magnitude2(nc)+(epsilon/100000))

def K_Means(expvecl,wlen,l=4,T=100):
	print("K_Means Called\n\n\n")
	cost =0
	oldCost =0
	dcost = 0
	dmean=0
	epsilon = 0.00001
	sigma = 0.001
	tau = 0.0001
	Matrix=[]
	for t in range(T):
		#generate L centers
		print("\t\t\tIteration t : %d"%(t))
		centers=[]
		for i in range(l):
			vec = []
			for j in range(wlen):
				vec.append(random.uniform(0, 300000))
			centers.append(vec)
		fcenters = copy.deepcopy(centers)
		Matrix = [[] for n in range(l)]
		print("\t\t\tTotal :",len(expvecl))
		for p,ev in enumerate(expvecl):
			print(p)
			disto = 1000000000
			ind = 0
			for x in range(l):
				dst = np.multiply.outer(centers[x],ev).ravel().sum()/(magnitude2(centers[x])*magnitude2(ev)+(epsilon/100000))
				if dst < disto:
					disto = dst
					ind = x
			Matrix[ind].append(p)
			cost += disto
			# Update the centers
		print("Updating the clusters")
		for x in range(l):
			fcenters[x] = addAll(Matrix[x],expvecl)
		dmean = 0
		for oc in centers:
			for nc in fcenters:
				dmean += vecF(oc,nc)
		#dmean = np.sum(F(np.transpose([np.tile(centers,len(fcenters)),np.repeat(fcenters,len(centers))])))
		dcost = math.fabs(cost-oldCost)/(oldCost+epsilon)
		oldCost = cost
		if dcost <= sigma or dmean <= tau:
			return Matrix
	return Matrix


if __name__ == "__main__":
	a = time.time()
	B = getBagOfWords("singletest.txt",a)
	ChunkArr = RepresentChunk(B,a,"singletest.txt")
	sense,wl,centroids = getSense(ChunkArr,a)
	print(wl,centroids)
	print(getConceptVecEntityList(sense[0]))
	b = time.time()
	print("It took %s seconds."%(b-a))


