#fetch data
doc='test.txt'
dataSet=[]
f_list = open(doc).readlines()
for f in f_list:
	dataSet.append(f)