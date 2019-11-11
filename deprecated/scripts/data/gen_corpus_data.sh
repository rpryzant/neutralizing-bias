# generate train/test splits from gen_data_from_crawl.py outputs

# sh gen_corpus_data.sh ../../data/v4/raw/tst.biased ../../data/v4/raw/tst.unbiased_for_full_mix ../../data/v4/full ../../data/v4/raw/vocab.bert
# sh gen_corpus_data.sh ../../data/v4/raw/tst.wordbiased ../../data/v4/raw/tst.unbiased_for_word_mix ../../data/v4/word ../../data/v4/raw/vocab.bert 


BIASED_CORPUS=$1
UNBIASED_CORPUS=$2
OUT_DIR=$3
VOCAB=$4 # BERT vocab etc
TEST_SIZE=2000

mkdir $OUT_DIR

# data is
# rev_id, prev_toks, post_toks, prev_raw, post_raw, sent_label, tok_labels

echo "SHUFFLING..."
cat $BIASED_CORPUS | gshuf > $OUT_DIR/biased.raw
cat $UNBIASED_CORPUS | gshuf > $OUT_DIR/unbiased.raw

echo "SPLITTING..."
tail -n +$TEST_SIZE $OUT_DIR/biased.raw > $OUT_DIR/biased.train
head -n $TEST_SIZE $OUT_DIR/biased.raw > $OUT_DIR/biased.test

echo "MIXING..."
cat $OUT_DIR/biased.train $OUT_DIR/unbiased.raw | gshuf > $OUT_DIR/mixed.train

echo "SEPERATING..."
cat $OUT_DIR/biased.train | cut -f2 > $OUT_DIR/biased.train.pre
cat $OUT_DIR/biased.train | cut -f3 > $OUT_DIR/biased.train.post
cat $OUT_DIR/biased.train | cut -f6 > $OUT_DIR/biased.train.seq_labels
cat $OUT_DIR/biased.train | cut -f7 > $OUT_DIR/biased.train.tok_labels

cat $OUT_DIR/mixed.train | cut -f2 > $OUT_DIR/mixed.train.pre
cat $OUT_DIR/mixed.train | cut -f3 > $OUT_DIR/mixed.train.post
cat $OUT_DIR/mixed.train | cut -f6 > $OUT_DIR/mixed.train.seq_labels
cat $OUT_DIR/mixed.train | cut -f7 > $OUT_DIR/mixed.train.tok_labels

cat $OUT_DIR/biased.test | cut -f2 > $OUT_DIR/biased.test.pre
cat $OUT_DIR/biased.test | cut -f3 > $OUT_DIR/biased.test.post
cat $OUT_DIR/biased.test | cut -f6 > $OUT_DIR/biased.test.seq_labels
cat $OUT_DIR/biased.test | cut -f7 > $OUT_DIR/biased.test.tok_labels

echo "MAKING VOCABS..."
cp $VOCAB $OUT_DIR/vocab
python make_attribute_vocab.py $OUT_DIR/vocab $OUT_DIR/biased.train.pre $OUT_DIR/biased.train.post 1.5 > $OUT_DIR/vocab.attribute
