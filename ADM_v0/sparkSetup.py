import os
import sys
import json
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext
from pyspark.sql.functions import col

class Probase:
	sc = None
	sqlContext = None
	def cls(self):
		os.system('cls' if os.name=='nt' else 'clear')
	
	def __init__(self):
		self.cls()
		conf = SparkConf().setAppName("ADM")
		self.sc = SparkContext(conf=conf)
		self.sqlContext = SQLContext(self.sc)

	def __del__(self):
		if self.sc != None:
			self.sc.stop()

	def isAConcept(self,word):
		#dfc = self.sqlContext.read.format("jdbc").options(url="jdbc:mysql://localhost:3306/probase",dbtable = word[0:2].upper()+"_Concept",user="root",password="conceptcluster").load()
		#dfe = self.sqlContext.read.format("jdbc").options(url="jdbc:mysql://localhost:3306/probase",dbtable = word[0:2].upper()+"_Concept",user="root",password="conceptcluster").load()
		#l = self.dfc.where(self.dfc._c0 == word).select("_c2").count() > self.dfe.where(self.dfe._c1 == word).select("_c2").count()
		self.dataframeC = self.sqlContext.read.format("csv").option("delimiter", "\t").load("data/"+word[0:2].upper()+"_Concept.txt")
		self.dataframeE = self.sqlContext.read.format("csv").option("delimiter", "\t").load("data/"+word[0:2].upper()+"_Instance.txt")
		l = self.dataframeC.where(self.dataframeC._c0 == word).select("_c2").count() > self.dataframeE.where(self.dataframeE._c1 == word).select("_c2").count()
		return l

if __name__=="__main__":
	p = Probase()
	start = time.time()
	print("Before call...")
	l = p.isAConcept("company")
	print(l)
	end = time.time()
	print("Call ended...")
	print(end-start)
	start = time.time()
	print("Before call...")
	l = p.isAConcept("microsoft")
	print(l)
	end = time.time()
	print("Call ended...")
	print(end-start)
	
