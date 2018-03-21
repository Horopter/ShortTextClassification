import re

class Document():
	line=""
	def __init__(self, s_line):
		self.line=s_line
		re.sub('[^A-Za-z0-9]+','',self.line)
		self.line = ''.join([x for x in self.line if ord(x) < 128])

