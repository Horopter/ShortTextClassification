from Sense import getSense
import time
import _pickle as cPickle
from pathlib import Path
from scipy import spatial
from scipy import linalg, mat, dot

SenseSet = []

SenseTrainNames = ["business","computer","health","politics","sports"]

def trainDataSet():
	sensedata = Path("SenseSet.pkl")
	if sensedata.is_file():
		SenseSet = cPickle.load(open("SenseSet.pkl","rb"))
		L1,D1,F1,M1,C1,Cn1,G1 = SenseSet[0]
		print(L1)
	else:
		businessSense = getSense("business.txt")
		print("Training model : 20%  done")
		computerSense = getSense("computer.txt")
		print("Training model : 40%  done")
		healthSense = getSense("health.txt")
		print("Training model : 60%  done")
		politicsSense = getSense("politics.txt")
		print("Training model : 80%  done")
		sportsSense = getSense("sports.txt")
		print("Training model : 100%  done")
		SenseSet = [businessSense,computerSense,healthSense,politicsSense,sportsSense]
		cPickle.dump(SenseSet,open("SenseSet.pkl","wb"))

def testDataSet():
	testSense = getSense("test.txt")
	L1,D1,F1,M1,C1,Cn1,G1 = testSense
	for i,xSense in enumerate(SenseSet):
		L2,D2,F2,M2,C1,Cn2,G2 = xSense
		L = intersect(L1,L2)
		if len(L)==0:
			L2 = [L2.index(i) for i in L2]
			L2 = [F2[i] for i in L2]
			L1 = [L1.index(i) for i in L1]
			L1 = [F1[i] for i in L1]
			L1 = mat(L1)
			L2 = mat(L2)
			Cos_Similarity = dot(L1,L2.T)/linalg.norm(L1)/linalg.norm(L2)
			print(type(Cos_Similarity))
			print("The given document is conceptually related to %s with a semantic distance of %f.\n"%(SenseTrainNames[i],1-Cos_Similarity));
		else:
			print("The given document is conceptually related to %s.\n"%(SenseTrainNames[i]));



if __name__ == "__main__":
	t = time.clock()
	trainDataSet()
	u = time.clock()
	print("You spent %d seconds."%(u-t))
	testDataSet()