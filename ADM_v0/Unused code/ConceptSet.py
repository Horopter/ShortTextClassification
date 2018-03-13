from Bag_Of_Words import getScores
from collections import Counter
import pymysql.cursors
from sklearn import metrics
import time

def getConceptSet(filename):
	print("For the datachunk : "+filename)
	print("Generating Concept set : Progress -> 10%")
	B = getScores(filename)
	print("Generating Concept set : Progress -> 40%")
	C = []
	Cdict = {}
	C_set = []
	for c in B.Concepts:
		C_set.extend(list(set(sorted(c.keys()))))
	C_set = sorted(list(set(C_set)))
	print("Generating Concept set : Progress -> 70%")
	# print(C_set)
	Matrix = [[0 for x in range(len(B.Concepts))] for y in range(len(C_set))]
	for i in range(len(B.Concepts)):
		for j in range(len(C_set)):
			if C_set[j] in B.Concepts[i]:
				Matrix[j][i] = B.Concepts[i][C_set[j]]
			else:
				Matrix[j][i]=0
	list1 = metrics.pairwise.cosine_similarity(X=Matrix, Y=None, dense_output=True)
	list1 = [[1-item for item in row]for row in list1]
	# print("Concept Vector is as : ")
	# print('\n'.join([' , '.join(['{:.4f}'.format(item) for item in row])for row in Matrix]))
	# print("Cosine Similarity is as : ")
	# print('\n\n'.join([' , '.join(['{:.4f}'.format(item) for item in row])for row in list1]))
	print("Generating Concept set : Progress -> 100%")
	return (C_set,list1,Matrix)

if __name__ == "__main__":
	start = time.clock()
	fv = getConceptSet("test.txt")
	print("It took %s seconds to get Concept set.",time.clock()-start)