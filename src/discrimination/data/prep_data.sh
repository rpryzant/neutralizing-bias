# sh prep_data.sh ../../../data/v2/corpus.biased.raw ../../../data/v2/corpus.unbiased.raw TEST

BIASED_CORPUS=$1
UNBIASED_CORPUS=$2
OUT_DIR=$3

mkdir $OUT_DIR

echo 'joining, shuffling...'
cat $BIASED_CORPUS $UNBIASED_CORPUS > $OUT_DIR/corpus.raw
cat $OUT_DIR/corpus.raw | gshuf > $OUT_DIR/corpus.shuf

echo 'prepping text...'
cat $OUT_DIR/corpus.shuf | cut -f3 > $OUT_DIR/corpus.text
tail -n +5000 $OUT_DIR/corpus.text > $OUT_DIR/text.train
head -n +5000 $OUT_DIR/corpus.text > $OUT_DIR/text.test

echo 'prepping labels...'
cat $OUT_DIR/corpus.shuf | cut -f5 > $OUT_DIR/corpus.labels
tail -n +5000 $OUT_DIR/corpus.labels > $OUT_DIR/labels.train
head -n +5000 $OUT_DIR/corpus.labels > $OUT_DIR/labels.test


