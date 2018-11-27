CORPUS=$1
OUT_DIR=$2

mkdir $OUT_DIR

cat $CORPUS | gshuf > $OUT_DIR/corpus.raw
cat $OUT_DIR/corpus.raw | cut -f3 > $OUT_DIR/pre
cat $OUT_DIR/corpus.raw | cut -f4 > $OUT_DIR/post

cat $OUT_DIR/pre $OUT_DIR/post > $OUT_DIR/all
python make_vocab.py $OUT_DIR/all 20000 > $OUT_DIR/vocab.20000
python make_attribute_vocab.py $OUT_DIR/vocab.20000 $OUT_DIR/pre $OUT_DIR/post > $OUT_DIR/vocab.attribute 1.3

tail -n +1000 $OUT_DIR/pre > $OUT_DIR/pre.train
tail -n +1000 $OUT_DIR/post > $OUT_DIR/post.train

head -n 1000 $OUT_DIR/pre > $OUT_DIR/pre.test
head -n 1000 $OUT_DIR/post > $OUT_DIR/post.test
