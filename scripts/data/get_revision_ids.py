# FROM HINOKI!
# python get_revision_ids.py /data/rpryzant/wiki/enwiki-20181120-stub-meta-history.xml


import sys
import xml.etree.cElementTree as ET
from tqdm import tqdm
import re

wiki_xml_path = sys.argv[1]

revisions = []


class Revision():
    def __init__(self):
        self.revid = None
        self.comment = None
        self.timestamp = None        

    def incomplete(self):
        return not self.revid or not self.comment or not self.timestamp 

    def is_admissible(self):
        c_lower = self.comment.lower()
        if 'reverted' in c_lower or 'undid' in c_lower:
            return False
        if 'pov' in c_lower or 'npov' in c_lower or 'neutral' in c_lower:
            return True
        return False

    def print_out(self):
        print('\t'.join([self.revid, self.comment, self.timestamp]))

cur_rev = Revision()
page_skip = False
for line in tqdm(open(wiki_xml_path), total=11325433847):
    line = line.strip()
    if line == '<page>':
        page_skip = False
    if "<title>>" in line and ('user:' in line.lower() or 'talk:' in line.lower()): 
        page_skip = True
    if page_skip:
        continue

    if line == '</revision>':
        if not cur_rev.incomplete() and cur_rev.is_admissible():
            cur_rev.print_out()
        cur_rev = Revision()

    elif '<id>' in line and cur_rev.revid is None:  # avoid comment id
            cur_rev.revid = re.sub('</?[\w]+>', '', line)
    elif '<comment>' in line:
        cur_rev.comment = re.sub('</?[\w]+>', '', line)
    elif '<timestamp>' in line:
        cur_rev.timestamp = re.sub('</?[\w]+>', '', line)




