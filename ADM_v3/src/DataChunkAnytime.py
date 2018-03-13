from Document import *

class DataChunkAny:
	DocList=[]
	ChunkName=""
	tm=0
	def __init__(self,doc,tm):#takes input as filename
		self.ChunkName=doc
		self.tm=tm
		f_list = open(doc).readlines()
		for f in f_list:
			self.DocList.append(Document(f))

	def size():
		return(len(self.DocList))
