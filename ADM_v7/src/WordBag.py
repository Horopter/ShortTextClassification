from DataChunk import *
import re
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
from nltk.corpus import stopwords
import nltk
from decimal import *
import os
import sys
import _pickle as cPickle
import re



#stopwords from NLTK english.txt
connection = pymysql.connect(host="localhost",user="root",password="conceptcluster",db="probase")

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return len(bloblist) / (1 + n_containing(word, bloblist)) #logarithm is not used as the chunk is very small

def tfidf(word, blob, bloblist):
    return float(Decimal(1+tf(word, blob)).log10() * Decimal(idf(word, bloblist)).log10())


class BagOfWords():
	AllConcepts=[]
	E_bloblist=[]
	C_bloblist=[]

	def __init__(self, dc=None):
		if dc is None:
			self.AllConcepts=[]
			self.E_bloblist=[]
			self.C_bloblist=[]
			return
		for d in dc.DocList:#break each doc into its entiites ,remove stopwords
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
	if len(letter) != 2:
		return False
	concept_count = 0
	entity_count = 0
	connection = pymysql.connect(host="localhost",user="root",password="conceptcluster",db="probase")
	try:
	    with connection.cursor() as cursor:
	        # Read a single record
	        sql = "SELECT SUM(`frequency`) FROM `"+letter.upper()+"_Concept` where Concept = \""+word+"\";"
	        #print(sql)
	        cursor.execute(sql)
	        result = cursor.fetchone()
	        #print(result)
	        concept_count = result[0]
	    with connection.cursor() as cursor:
	        # Read a single record
	        sql = "SELECT SUM(`frequency`) FROM `"+letter.upper()+"_Instance` where entity = \""+word+"\";"
	        #print(sql)
	        cursor.execute(sql)
	        result = cursor.fetchone()
	        #print(result)
	        entity_count = result[0]
	finally:
	    connection.close()
	if concept_count == None: concept_count = 0
	if entity_count == None: entity_count = 0
	return float(concept_count) > float(entity_count)

def isAConcept(word):
	p = Probase()
	return p.isAConcept(word)

def getConcepts(word):
	letter = word[0:2]
	re.sub("[^A-Za-z]+","",letter)
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

# def getConcepts(word):# Return the top 5 concepts
# 	with requests.Session() as s:
# 		r=s.get("https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance="+word+"&topK=5")
# 		data=r.json()
# 		d = {}
# 		for k,v in data.items():
# 			l = []
# 			l.append(v)
# 			d[k] = l
# 		return dict(data)

def getBagOfWords(F,start):
	if not (os.path.isfile(F+"-Bag.pkl")):
		print("\tCommencing Bag Representation for %s "%(F))
		D=DataChunk(F)
		print("\tDataChunk Representation Completed.")
		B=BagOfWords(D)
		print("\tWordBag gathered.")
		end = time.time()
		print("\tBag Representation Completed at %s seconds"%(end-start))
		cPickle.dump((B.E_bloblist,B.C_bloblist,B.AllConcepts),open(F+"-Bag.pkl","wb"))
	else:
		B=BagOfWords()
		B.E_bloblist,B.C_bloblist,B.AllConcepts = cPickle.load(open(F+"-Bag.pkl","rb"))
	return B

if __name__ == "__main__":
	a = time.time()
	B = getBagOfWords("test.txt",a)
	b = time.time()
	#print(B.E_bloblist)
	#print("\n\n")
	#print(B.C_bloblist)
	#print("\n\n")
	#print(B.AllConcepts)
	print(b-a)

