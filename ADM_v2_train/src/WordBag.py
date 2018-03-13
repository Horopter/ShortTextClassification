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

stopwords = {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"}
#stopwords from NLTK english.txt
connection = pymysql.connect(host="localhost",user="root",password="conceptcluster",db="probase")

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

	def __init__(self, dc):
		s=requests.session()
		for d in dc.DocList:#break each doc into its entiites ,remove stopwords
			candidate=re.sub("[^\w]", " ",  d.line).split()#get document line by line
			ent=[]
			con=[]
			for c in candidate:
				if c in stopwords: 
					continue
				else:
					if isConcept(c):
						con.append(c)
					else:
						ent.append(c)
			ent=list(set(ent))#repetitions are not significant in short text
			uselessEnt=[]
			EtoConPerDoc=[]
			for e in ent:
				enTocon=getConcepts(e,s)
				enTocon = list(enTocon.keys())
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
	global connection
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
	    pass
	if concept_count == None: concept_count = 0
	if entity_count == None: entity_count = 0
	return concept_count > entity_count

def isAConcept(word):
	p = Probase()
	return p.isAConcept(word)

def getConcepts(word,s):# Return the top 5 concepts
	r=s.get("https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance="+word+"&topK=5")
	data=r.json()
	d = {}
	for k,v in data.items():
		l = []
		l.append(v)
		d[k] = l
	return dict(data)

def getBagOfWords(F,start):
	print("\tCommencing Bag Representation for %s "%(F))
	D=DataChunk(F)
	print("\tDataChunk Representation Completed.")
	B=BagOfWords(D)
	print("\tWordBag gathered.")
	end = time.time()
	print("\tBag Representation Completed at %s seconds"%(end-start))
	return B

if __name__ == "__main__":
	a = time.time()
	B = getBagOfWords("business-sample.chunk",a)
	b = time.time()
	#print(B.E_bloblist)
	#print("\n\n")
	#print(B.C_bloblist)
	#print("\n\n")
	#print(B.AllConcepts)
	print(b-a)

