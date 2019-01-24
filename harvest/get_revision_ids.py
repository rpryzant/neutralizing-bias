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
        # negative filter on revisions
        self.INVALID_REV_RE = 'revert|undo|undid|robot'   
        # NPOV detector. Essentially looks for common pov-related words
        #     pov, depov, npov, yespov, attributepov, rmpov, wpov, vpov, neutral
        # with certain leading punctuation allowed
        self.NPOV_RE = '([- wnv\/\\\:\{\(\[\"\+\'\.\|\_\)\#\=\;\~](rm)?(attribute)?(yes)?(de)?n?pov)|([- n\/\\\:\{\(\[\"\+\'\.\|\_\)\#\;\~]neutral)'

    def incomplete(self):
        return not self.revid or not self.comment or not self.timestamp

    def is_admissible(self):
        c_lower = self.comment.lower()


        if re.search(self.INVALID_REV_RE, c_lower):
            return False
        if re.search(self.NPOV_RE, c_lower):
            if 'pover' in c_lower: # special case: "poverty", "impovershiment", etc
                return False
            return True
        return False

    def print_out(self):
        print('\t'.join([self.revid, self.comment, self.timestamp]))

SPECIAL_TITLE_RE = "<title>.*?(talk|user|wikipedia)\:"

cur_rev = Revision()
page_skip = False
for line in tqdm(open(wiki_xml_path), total=11325433847):
    line = line.strip()
    line_lower = line.lower()
    if line == '<page>':
        page_skip = False
    if re.search(SPECIAL_TITLE_RE, line_lower):
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
        cur_rev.comment = re.sub('</?[\w]+>', '', line).replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    elif '<timestamp>' in line:
        cur_rev.timestamp = re.sub('</?[\w]+>', '', line)




