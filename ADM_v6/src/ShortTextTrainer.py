from ClusterHelper import *
from ConceptCluster import *
from ConceptRep import *
from AnyConceptRep import *
from AnyWordBag import getAnyBagOfWords
from WordBag import *
import time
import _pickle as cPickle
import os.path
import string 
from collections import Counter
import multiprocessing
import itertools
import operator
import sys

"""
Document Cluster example:

DocClu:  
[
	[
		(0, 
			[
				('plus', {'issue': -0.0, 'online': 763.7223651432431, 'websites': -107.91780938291092, 'clinics': 234.9640504830937, 'instructions': 2143.1671406272517, 'aftercare': 1057.5931403238658}), ('procedures', {'issue': -606.2556558354908, 'online': -1231.5589070911874, 'websites': -0.0, 'clinics': 336.87866565334883, 'instructions': 182.8964809758706, 'aftercare': 1255.8007628187856}),
				('service', {'issue': -2120.8925644025, 'online': -1182.5229897743834, 'websites': 640.970845542025, 'clinics': -290.92823485457706, 'instructions': -0.0, 'aftercare': -138.6094948760576})
			]
		),
		(1, 
			[
				('factor', {'issue': 2604.378173358898, 'online': -1212.1317978395805, 'websites': -406.915025054653, 'clinics': -76.9298877807952, 'instructions': -267.93850166214054, 'aftercare': -781.8942150869599}),
				('patient', {'issue': 1465.0490454734206, 'online': -300.62174361352317, 'websites': 182.8964809758706, 'clinics': -242.23992894311232, 'instructions': -1525.3888065765589, 'aftercare': -633.3360992037609}), 
				('providers', {'issue': -0.0, 'online': -0.0, 'websites': -361.129477484063, 'clinics': 335.73880949101795, 'instructions': -531.3772820703042, 'aftercare': -1527.089394544018})
			]
		)
	],
	[
		(0, 
			[
				('area', {'health': -336.7032771776612, 'medicine': -0.0, 'dir': 357.5913745099014, 'yahoo': 283.00325191700296, 'surgery': 1199.9326799454934, 'sit': 637.2376115639128}), 
				('command', {'health': -228.71569722464272, 'medicine': -27.273205405200677, 'dir': 122.10341204711564, 'yahoo': -457.2269104187375, 'surgery': 705.1441913658095, 'sit': 362.2479799736035}), 
				('medicine', {'health': -40.4406880000211, 'medicine': -200.95387624660543, 'dir': -441.1649431105886, 'yahoo': -1251.7598887770991, 'surgery': -1024.3041788753583, 'sit': -78.38420613251597}), 
				('review', {'health': -50.423706567964246, 'medicine': 773.4313730081977, 'dir': -415.03044349437823, 'yahoo': -69.23689900271567, 'surgery': 130.67891128564148, 'sit': -490.4175012266003}), 
				('surgery', {'health': -170.59904061306904, 'medicine': -955.3162908973172, 'dir': -0.0, 'yahoo': 717.5047282412103, 'surgery': 257.7620627258861, 'sit': 169.99663174258478})
			]
		), 
		(1, 
			[
				('directory', {'health': -0.0, 'medicine': 65.50557437354765, 'dir': -304.431310955309, 'yahoo': 822.9287593778438, 'surgery': -1223.9243441557226, 'sit': -524.5390046381668})
			]
		)
	]
]

ExpVecL on DocClu:

[
	[
		('aftercare', -255.84510018938147),
		('clinics', 99.1611580163253),
		('dir', 0),
		('health', 0),
		('instructions', 0.4530104313727179),
		('issue', 447.4263328647759),
		('medicine', 0),
		('online', -1054.3710243918104),
		('sit', 0),
		('surgery', 0),
		('websites', -17.3649951345771),
		('yahoo', 0)
	], 
	[
		('aftercare', 0),
		('clinics', 0),
		('dir', -379.731430964899),
		('health', -165.37648191667168),
		('instructions', 0),
		('issue', 0),
		('medicine', -16.516825534637476),
		('online', 0),
		('sit', -404.4029014539699),
		('surgery', -970.0816108662282),
		('websites', 0),
		('yahoo', 667.385615769776)
	]
]

"""


def getChunkCluster(F,start):
	B = getBagOfWords(F,start)
	ChunkArr = RepresentChunk(B,start,F)
	cs,wl = getSense(ChunkArr,start)
	print("\t\tInitializing Document Cluster setup...")
	wl = sorted(wl)
	wlen = len(wl)
	expvecl = []
	for i in cs:
		V,el = getConceptVecEntityList(i)
		expvec = getExpandedVector(V,el,wl)
		expvecl.append([v for k,v in expvec])
	print("\t\tExpanded the vector space.\n\t\tRunning 100 iterations to find optimal cluster centers.")
	km = K_Means(expvecl,wlen,3,100)
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

def StartTraining(filenames,start):
	nb_cpus = 4
	pool = multiprocessing.Pool(processes=nb_cpus)
	pool.map(TrainChunk, [(a, start) for a in filenames])
	TE = TrainEnsembleSeq((filenames,start))
	return TE

def driver(start):
	try:
		print(time.ctime())
		filenames = ['engineering.chunk', 'health.chunk','computers.chunk','business.chunk']
		#SInce this pickle has only 5 clusters
		#timeToProcess=random.randint(200,400)
		TE = StartTraining(filenames,start)
	finally:
		print(time.ctime())

if __name__=="__main__":
	#docClu = None #copy from above
	#getDocumentClusterCenter(docClu)
	driver(time.time())
	