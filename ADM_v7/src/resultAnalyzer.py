import re
import string

def parseResultText(fname):
	content = []
	chars = re.escape(string.punctuation)
	with open(fname, encoding='utf-8', errors='ignore') as f:
		content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content]
		content = [x.replace("Rep_test_","").replace(".chunk","").replace(" : Final Result is expected = ","|").replace(" and given is ","|").replace("_","|").replace(".","") for x in content]
		content = [x.split("|") for x in content]
		content = [[x[0],x[2],x[3]] for x in content if len(x)==4]
		content = [[x[0],x[1],re.sub(' +',' ',re.sub(r'['+chars+']', ' ',x[2])).strip()] for x in content]
		content = [[x[0],x[1],x[2].split(" ")] for x in content]
	return content

def genConfusionMatrix(parsedList):
	confusionMatrix = []
	for lis in parsedRes:
		number = lis[0]
		expectedResult = lis[1]
		givenResults = lis[2]
		if expectedResult == givenResults[0]:
			confusionMatrix.append((number,"Accurate"))
		elif expectedResult in givenResults:
			confusionMatrix.append((number,"Guessed"))
		else:
			confusionMatrix.append((number,"Wrong"))
	return confusionMatrix

if __name__=="__main__":
	parsedRes = parseResultText('results.txt')
	cm = genConfusionMatrix(parsedRes)
	cm = sorted(cm, key=lambda x: int(x[0]))
	print("Confusion Matrix : \n",cm)
	print("Confusion Matrix Length : \n",len(cm))
	accurate = len([x for x in cm if x[1] == "Accurate"])
	guessed = len([x for x in cm if x[1] == "Guessed"])
	wrong = len([x for x in cm if x[1] == "Wrong"])
	accuratepct = accurate*100/len(cm)
	guessedpct = guessed*100/len(cm)
	wrongpct = wrong*100/len(cm)
	print(accuratepct,guessedpct,wrongpct)
