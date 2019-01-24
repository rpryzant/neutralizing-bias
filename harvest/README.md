
1. Download the latest dump of historical Wikipedia revision data.
```
wget https://dumps.wikimedia.org/enwiki/20190120/enwiki-20190120-stub-meta-history.xml.gz
export $WIKI_DATA=enwiki-20190120-stub-meta-history.xml.gz
```

2. Get all NPOV-related revision ids.
```
python get_revision_ids.py $WIKI_DATA > revisions.raw
```