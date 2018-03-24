from DataChunk import *
import re
import os.path
import math
import time
from Document import *
from textblob import TextBlob as tb
import collections
from nltk.stem.snowball import SnowballStemmer
import requests
import pymysql.cursors
from itertools import chain
from collections import defaultdict
from sparkSetup import *
import random
from nltk.corpus import stopwords
import nltk


#stopwords from NLTK english.txt


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return len(bloblist) / (1 + n_containing(word, bloblist)) #logarithm is not used as the chunk is very small

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


class BagOfWords():
	AllConcepts=[]
	E_bloblist=[]
	C_bloblist=[]

	def __init__(self, dc,tm):
		#s=requests.session()
		startTime=time.time()
		timeToProcess=tm/5 #Bag of words takes 1/5th of the total time for classification (Experimentally seen)
		#random.shuffle(dc.DocList) #Kind of random sampling in case time expires
		for d_i,d in enumerate(dc.DocList):#break each doc into its entiites ,remove stopwords
			if ( (time.time()-startTime) > timeToProcess):
				print("BagofWords Time Exceeded:\nProcessed: ",d_i," Documents")
				break

			candidate=re.sub("[^\w]", " ",  d.line).split()#get document is one line
			ent=[]
			con=[]
			candidate = [word for word in candidate if word not in stopwords.words('english')]
			candidate=[nltk.stem.WordNetLemmatizer().lemmatize(word,'v') for word in candidate]
			for c in candidate:
				
				if isConcept(c):
					con.append(c)
				else:
					ent.append(c)

			ent=list(set(ent))#repetitions are not significant in short text
			uselessEnt=[]
			EtoConPerDoc=[]
			for e in ent:
				enTocon=getConcepts(e)
				# enTocon = list(enTocon.keys())
				if not enTocon:
					uselessEnt.append(e)
				else:
					EtoConPerDoc+=enTocon
			lister = []
			for l in EtoConPerDoc:
				lister.append(l)
			Cdict = dict(collections.Counter(lister))
			Cdict = {k: v for k, v in Cdict.items() if v > 1} #Noisy concepts with only one occurrence are filtered. Page 5 of the paper
			EtoConPerDoc = list(Cdict.keys())

			ent=list(set(ent)-set(uselessEnt))

			EntoCon=list(set(EtoConPerDoc+list(set(con))))

			self.E_bloblist.append(tb(' '.join(ent)))
			self.C_bloblist.append(tb(' '.join(con)))
			self.AllConcepts.append(EntoCon)#extend would raise the cost of co,putation about 10 fold.
		
def isConcept(word):
	letter = word[0:2]
	letter = re.sub("[^A-Za-z]+","",letter)
	if len(letter) != 2:
		return False
	concept_count = 0
	entity_count = 0
	connection = pymysql.connect(host="localhost",user="root",password="conceptcluster",db="probase")
	try:
	    with connection.cursor() as cursor:
	        # Read a single record
	        sql = "SELECT SUM(`frequency`) FROM `"+letter.upper()+"_Concept` where Concept = \""+word+"\""
	        cursor.execute(sql)
	        result = cursor.fetchone()
	        concept_count = result[0]
	    with connection.cursor() as cursor:
	        # Read a single record
	        sql = "SELECT SUM(`frequency`) FROM `"+letter.upper()+"_Instance` where entity = \""+word+"\""
	        cursor.execute(sql)
	        result = cursor.fetchone()
	        entity_count = result[0]
	finally:
	    connection.close()
	if concept_count == None: concept_count = 0
	if entity_count == None: entity_count = 0
	return concept_count > entity_count

def isAConcept(word):
	p = Probase()
	return p.isAConcept(word)

def getConcepts(word):
	letter = word[0:2]
	letter = re.sub("[^A-Za-z]+","",letter)
	if len(letter) != 2:
		return False
	connection = pymysql.connect(host="localhost",user="root",password="conceptcluster",db="probase")
	try:
	    with connection.cursor() as cursor:
	        # Read a single record
	        sql = "SELECT Concept FROM `"+letter.upper()+"_Instance` where entity = \""+word+"\" order by popularity desc limit 10;"
	        print(sql)
	        cursor.execute(sql)
	        result = [item[0] for item in cursor.fetchall()]
	        return result
	finally:
	    connection.close()

def getAnyBagOfWords(F,start,TP):
	#Set random time limit to process the document chunk
	D=DataChunk(F)
	B=BagOfWords(D,TP)
	end = time.time()
	print("Bag Representation Complete at %s seconds"%(end-start))
	return B

if __name__ == "__main__":
	a = time.time()
	B = getBagOfWords("test.txt")#FOR CLASSIFICATION ie. TEST DOCUMENT CHUNK NAME
	b = time.time()
	#print(B.E_bloblist)
	#print("\n\n")
	#print(B.C_bloblist)
	#print("\n\n")
	#print(B.AllConcepts)
	print(b-a)

