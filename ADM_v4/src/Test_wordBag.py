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
from nltk.corpus import stopwords
import nltk

class BagOfWords():
	AllConcepts=[]
	E_bloblist=[]
	C_bloblist=[]

	def __init__(self, dc):
		for d in dc.DocList:#break each doc into its entiites ,remove stopwords
			candidate=re.sub("[^\w]", " ",  d.line).split()#get document line by line
			candidate = [word for word in candidate if word not in stopwords.words('english')]
			candidate=[nltk.stem.WordNetLemmatizer().lemmatize(word,'v') for word in candidate]
			print (candidate)


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