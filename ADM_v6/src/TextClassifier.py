from ClusterHelper import *
from ConceptCluster import *
from ConceptRep import *
from AnyConceptRep import *
from AnyWordBag import getAnyBagOfWords
from WordBag import *
import time
import os
import os.path
import string 
import multiprocessing
import itertools
import operator
import sys
import random
from ShortTextTrainer import StartTraining

def getFilePath(filename):
	this_dir=os.path.dirname(os.path.realpath(__file__))
	print("Cur dir :",this_dir)
	filePath = this_dir+"/Data/Test/"+filename
	return filePath

def second_highest(numbers):
    m1, m2 = float('-inf'), float('-inf')
    for x in numbers:
        if x >= m1:
            m1, m2 = x, m1
        elif x > m2:
            m2 = x
    return m2

def common(L,function):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return function(groups, key=_auxfun)[0]

def shortTextToChunkClusterMatch(coc,st):
	tcost = 0
	ind = -1
	costs=[]
	for j,docClu in coc.items():
		#print("DocClu: ",docClu)
		#print("ST : ",st)
		if docClu and st:
			cost=ShortTextChunkSemDistance(docClu,st)
		else:
			cost = 0
		costs.append((j,cost))
		#print("cost: ",cost," len(docClu) ",len(docClu),"\n\n\n")
		if cost > tcost:
			tcost = cost
			ind = j
	#print("Document Cluster Index = ",ind,"\n\n\n")
	#print(costs)
	return tcost

def ClassifyChunk(F,Flist,ChunkClusters,TP,start):
	B = getAnyBagOfWords(F,start,TP)
	ChunkArr = AnyRepresentChunk(B,start,4*TP/5)
	stl,wl = getSense(ChunkArr,start)
	ann=[]
	costsList=[]
	for st in stl:
		annotation=[]
		costs= []
		for x in ChunkClusters:
			cost = shortTextToChunkClusterMatch(x,st)
			costs.append(cost)
		print(costs)
		for en,c in enumerate(costs):
			if c == 0:
				annotation.append("unclassifiable")
			else:
				#print("Min cost and index are : ",min(costs)," ",costs.index(min(costs))," ")
				annotation.extend([Flist[i] for i, x in enumerate(costs) if x == max(costs)])#Let all guys get equal chance.
		#print("Annotation : ",annotation)
		annotation = list(filter(("unclassifiable").__ne__, annotation))
		if len(annotation) == 0 : annotation.append("unclassifiable")
		ann.append(common(annotation,max))
		costsList.append(costs)
	semsim = map(sum, zip(*costsList))
	print("Document wise classification :",ann)
	secChoice = [Flist[i] for i, x in enumerate(costs) if x == max(costs)]
	chunkClassification = "unclassifiable"
	if len(ann) > 0:
		chunkClassification = common(ann,max)
	return (chunkClassification.replace(".chunk",""),secChoice)

def driver(start):
	if os.path.isfile("ChunkClusters.pkl"):
		os.remove("ChunkClusters.pkl")
	train = ['engineering.chunk', 'health.chunk','computers.chunk','business.chunk']
	test = os.listdir("Data/Test")
	random.shuffle(test)
	matrix = []
	if not (os.path.isfile("ChunkClusters.pkl")):
		TE = StartTraining(train,start)
	os.system('tput clear')
	TestLen = len(test)
	x = random.choice(test)
	path = getFilePath(x)
	print("Path of test file is : ",path)
	timeToProcess=100
	print("GIVEN TIME TO PROCESS: ",timeToProcess," secs")
	number,expectedResult = x.replace("Rep_test_","").replace(".chunk","").split("_")
	result = ClassifyChunk(path,train,TE,timeToProcess,start)
	print("Classification for the chunk : ",result,expectedResult)
	print("Final Result is expected = %s and given is %s."%(expectedResult,result))

if __name__=="__main__":
	driver(time.time())