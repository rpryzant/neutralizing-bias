import csv, sys, os, re #, urllib2
from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen

# colors
class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def html2diff(html):
	prev_doc, next_doc = [],[]
	soup = BeautifulSoup(html, 'html')
	nodes = soup.find_all(class_=re.compile(r'(diff-deletedline)|(diff-addedline)|(diff-empty)'))
	div_p = re.compile(r'<div.*?>(.*)</div>', re.DOTALL)

	for i in range(int(len(nodes)/2)):
		node_prev = nodes[i*2]
		node_next = nodes[i*2+1]
		if node_prev.div:
			m1 = re.match(div_p, node_prev.div.prettify(formatter=None))
			if m1:
					prev_doc.append(m1.group(1).strip())
			else:
					prev_doc.append('')
		else:
			prev_doc.append('')
		if node_next.div:
			m2 = re.match(div_p, node_next.div.prettify(formatter=None))
			if m2:
					next_doc.append(m2.group(1).strip())
			else:
					next_doc.append('')
		else:
			next_doc.append('')
	return prev_doc, next_doc

def url2diff(url):
	try:
		response = urlopen(url)
		html = response.read()
		return html2diff(html)
	except Exception as e:
		print(e.reason, file=sys.stderr)


def print_withcolor(idx, l):
	l = l.replace('\n', '')
	ins_p = re.compile(r'<ins.*?>(.*?)</ins>', re.DOTALL)
	del_p = re.compile(r'<del.*?>(.*?)</del>', re.DOTALL)
	patterns = [ins_p, del_p]
	for i,p in enumerate(patterns):
		match = re.finditer(p, l)
		if match:
			new_l = ""
			last = 0
			for m in match:
				if i == 1:
					color = bcolors.OKBLUE
				else:
					color = bcolors.OKGREEN
				new_l = new_l + l[last:m.start(1)] + color + m.group(1) + bcolors.ENDC
				last = m.end(1)
			new_l = new_l + l[last:]
			l = new_l
	print (bcolors.HEADER+'line '+str(idx+1)+':'+bcolors.ENDC+l)


