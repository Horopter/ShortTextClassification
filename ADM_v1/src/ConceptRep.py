from WordBag import *
import pymysql.cursors
from math import log
from decimal import *
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
import numpy as np
import time

#Chunk representation per the paper
#We won't be using this for computations. It's just there as a prelude.

connection = pymysql.connect(host="localhost",user="root",password="conceptcluster",db="probase")

def representChunk(C):
	B = getBagOfWords(C)
	ChunkFV = []
	for i,blob in enumerate(B.E_bloblist):#for each document
		DocFV = []
		for concept in B.AllConcepts: # for each Concept
			EL = []
			for word in blob.words:#for each word
				val = tfidf(word, blob, B.E_bloblist)*float(pce(concept,word))*-1
				EL.append((word, val))
			DocFV.append({concept:EL})
		ChunkFV.append((DocFV,tupleMatrix))
	return ChunkFV

def getTupleMatrix(B):
	MatrixList=[]
	for i,blob in enumerate(B.E_bloblist):#for each document
		sortedWordList = sorted([word for word in blob.words])
		sortedConceptList = sorted([cncpt for cncpt in B.AllConcepts[i]])#B.AllConcepts could have been taken but it is redundant and time taking : leads to 2080 seconds per sense representation
		#print(sortedWordList)
		#print(sortedConceptList)
		Matrix = [[0 for x in range(len(sortedWordList))] for y in range(len(B.AllConcepts[i]))]
		for i,word in enumerate(sortedWordList):#for each word
			for j,concept in enumerate(sortedConceptList): # for each Concept
				val = tfidf(word, blob, B.E_bloblist)*float(pce(concept,word))*-100 # concept with minimum weight is more likely to be associated with the word.
				if val == float('inf') : val = 1000000
				Matrix[j][i] = val
		MatrixList.append((sortedWordList,sortedConceptList,Matrix))
	return MatrixList


def DistanceVector(M):
	cos_sim = metrics.pairwise.cosine_similarity(X=M, Y=None, dense_output=True)
	dm = [[1-item for item in row]for row in cos_sim]
	return dm


def pce(concept,word):
	letter = concept[0:2]
	concept_count = 0
	entity_occurrence = 0
	global connection
	try:
	    with connection.cursor() as cursor:
	        # Read a single record
	        sql = "SELECT SUM(`frequency`) FROM `"+letter.upper()+"_Concept` where Concept like \""+concept+"%\""
	        cursor.execute(sql)
	        result = cursor.fetchone()
	        concept_count = result[0]
	    with connection.cursor() as cursor:
	        # Read a single record
	        sql = "SELECT SUM(`frequency`) FROM `"+letter.upper()+"_Concept` where entity like \"%"+word+"%\" and Concept like \""+concept+"%\""
	        cursor.execute(sql)
	        result = cursor.fetchone()
	        entity_occurrence = result[0]
	finally:
	    # connection.close()
	    pass
	if concept_count == None: concept_count = 0
	if entity_occurrence == None: entity_occurrence = 0
	return Decimal(entity_occurrence/concept_count).log10()

def RepresentChunk(F):
	start = time.time()
	print("Starting Chunk representation at %s seconds"%(time.time()-start))
	B = getBagOfWords(F)
	print("Starting TupleMatrix computation at %s seconds"%(time.time()-start))
	TM=getTupleMatrix(B)
	print("Ended TupleMatrix computation at %s seconds"%(time.time()-start))
	ChunkRep = []
	for tm in TM:
		print("Starting document computation at %s seconds"%(time.time()-start))
		wl,cl,matrix = tm
		print("m is :",matrix,"\n")
		dv = DistanceVector(matrix)
		for i,l in enumerate(dv):
			for j,m in enumerate(dv[i]):
				dv[i][j] = 10000*dv[i][j]
				if Decimal(dv[i][j]) < Decimal(0.0001):
					dv[i][j] = 0
			#print(dv[i],"\n")
		ChunkRep.append((wl,cl,matrix))
		print("Ending document computation at %s seconds"%(time.time()-start))
	end = time.time()
	print("Chunk Representation Complete at %s seconds"%(end-start))
	return ChunkRep


if __name__=="__main__":
	RepresentChunk("singletest.txt")