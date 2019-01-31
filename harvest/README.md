
1. Download the latest dump of historical Wikipedia revision data.
```
wget https://dumps.wikimedia.org/enwiki/20190120/enwiki-20190120-stub-meta-history.xml.gz
gunzip enwiki-20190120-stub-meta-history.xml.gz
export $WIKI_DATA=enwiki-20190120-stub-meta-history.xml
```

2. Get all NPOV-related revision ids.
```
python get_revision_ids.py $WIKI_DATA > revisions.ids
```


3. Download the diffs for these revision ids. Note that this can take some time (300+ hours) so you may want to split into multiple jobs and join the outputs:

```
split -n 8 -d revisions.ids
python crawl_revision_text.py x00 > y00
            ...
python crawl_revision_text.py x07 > y07
cat y* > revisions.text
```

4. Clean, filter, and tag the revisions:

```
python gen_data_from_crawl.py revisions.text CACHE corpus
```

5. Add POS and Dependancy tags to the training data.

```
python add_tags.py corpus.wordbiased > corpus.wordbiased.tag
```

6. Shuffle and split the train data. Shuffle the unbiased data.

```
sh shuffle_split.sh corpus.wordbiased.tag
cat corpus.unbiased | gshuf > corpus.unbiased.shuf
```

