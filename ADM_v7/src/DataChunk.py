from Document import *

class DataChunk:
	DocList=[]
	ChunkName=""
	def __init__(self,doc):#takes input as filename
		self.ChunkName=doc
		f_list = open(doc).readlines()
		for f in f_list:
			self.DocList.append(Document(f))

	def size():
		return(len(self.DocList))
