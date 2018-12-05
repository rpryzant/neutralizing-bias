BIASED_CORPUS=$1
UNBIASED_CORPUS=$2
OUT_DIR=$3

mkdir $OUT_DIR

echo "SHUFFLING..."
cat $BIASED_CORPUS | gshuf > $OUT_DIR/corpus.biased.raw
cat $OUT_DIR/corpus.biased.raw | cut -f3 > $OUT_DIR/pre.biased
cat $OUT_DIR/corpus.biased.raw | cut -f4 > $OUT_DIR/post.biased

cat $UNBIASED_CORPUS | gshuf > $OUT_DIR/corpus.unbiased.raw
cat $OUT_DIR/corpus.unbiased.raw | cut -f3 > $OUT_DIR/pre.unbiased
cat $OUT_DIR/corpus.unbiased.raw | cut -f4 > $OUT_DIR/post.unbiased

echo "MAKING VOCAB..."
cat $OUT_DIR/pre.biased $OUT_DIR/pre.unbiased $OUT_DIR/post.biased $OUT_DIR/post.unbiased > $OUT_DIR/all
python make_vocab.py $OUT_DIR/all 24000 > $OUT_DIR/vocab.24000

echo "MAKING ATTRIBUTE VOCAB..."
python make_attribute_vocab.py $OUT_DIR/vocab.24000 $OUT_DIR/pre.biased $OUT_DIR/post.biased 1.3 > $OUT_DIR/vocab.attribute

echo "MAKING SPLITS..."
tail -n +700 $OUT_DIR/pre.biased > $OUT_DIR/pre.biased.train
tail -n +700 $OUT_DIR/post.biased > $OUT_DIR/post.biased.train

head -n 700 $OUT_DIR/pre.biased > $OUT_DIR/pre.biased.test
head -n 700 $OUT_DIR/post.biased > $OUT_DIR/post.biased.test

echo "MIXING BIASED/UNBIASED FOR 2nd TRAIN SET"
cat $OUT_DIR/pre.unbiased $OUT_DIR/pre.biased.train > $OUT_DIR/pre.mixed
cat $OUT_DIR/post.unbiased $OUT_DIR/post.biased.train > $OUT_DIR/post.mixed
paste $OUT_DIR/pre.mixed $OUT_DIR/post.mixed > $OUT_DIR/both.mixed
cat $OUT_DIR/both.mixed | gshuf > $OUT_DIR/both.mixed.shuf
cat $OUT_DIR/both.mixed.shuf | cut -f1 > $OUT_DIR/pre.mixed.train 
cat $OUT_DIR/both.mixed.shuf | cut -f2 > $OUT_DIR/post.mixed.train 