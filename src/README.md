
## Overview

See [shared/args.py](https://github.com/rpryzant/neutralizing-bias/blob/master/src/shared/args.py) for a complete list of CLI args.

This directory has the following subdirectories:
* `lexicons/`: text files with expert features from the literature (basically word lists).
* `tagging/`: The tagging model (it tags biased words in a sentence)
* `seq2seq/`: The CONCURRENT model (it converts biased sentences into neutral form)
* `joint/`: The MODULAR model (it has both tagger and CONCURRENT models inside of it) 
* `shared/`: Code that is common to all components: data iterators, command line arguments, constants, and beam search

Each model directory (`tagging`, `seq2seq`, `joint`) has the following files:
* `model.py`: modeling code (whether that be the tagger, seq2seq, or joint model)
* `utils.py`: code for (1) training and (2) evaluation 
* `train.py`: main driver code that builds and trains a model, evaluating after each epoch. All files take the same command line arguments (see `shared/args.py` for a list of arguments).  TODO LINK

To run any of the commands given below, you must first do the following:

(1) Download and unpack [the data](http://bit.ly/bias-corpus).

(2) `$ export DATA=/path/to/bias_data/WNC/`


## Run Tests

`$ sh integration_test.sh`

## Run Tagger Model

### Train

```
python tagging/train.py \
	--train $DATA/biased.word.train \
	--test $DATA/biased.word.test \
	--categories_file $DATA/revision_topics.csv \
	--extra_features_top --pre_enrich --activation_hidden --category_input \
	--learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 32 \
	--test_batch_size 16 --debias_weight 1.3 --working_dir train_tagging/
```
Takes ~40 minutes on a TITAN X gpu.

## Run Concurrent Model

### Train

```
python seq2seq/train.py \
       --train $DATA/biased.word.train \
       --test $DATA/biased.word.test \
       --pretrain_data $DATA/unbiased \
       --bert_full_embeddings --bert_encoder --debias_weight 1.3 \
       --pointer_generator --coverage --no_tok_enrich \
       --working_dir train_concurrent/
```

Checkpoints, tensorboard summaries, and per-epoch evaluations and decodings will go in your working directory. Takes ~25 hours on a TITAN X gpu.


### Inference

```
python joint/inference.py \
       --test $DATA/biased.word.test \
       --bert_full_embeddings --bert_encoder --debias_weight 1.3 \
       --pointer_generator --coverage --no_tok_enrich \  # no_tok_enrich makes it run as a seq2seq
       --working_dir inference_concurrent/ \ 
       --inference_output inference_concurrent/output.txt \
       --debias_checkpoint train_concurrent/model_X.ckpt
```

Evaluations and decodings will go in your working directory. 



## Run Modular Model

### Train

```
python joint/train.py \
       --train $DATA/biased.word.train \
       --test $DATA/biased.word.test \
       --pretrain_data $DATA/unbiased \
       --categories_file $DATA/revision_topics.csv --category_input \
       --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
       --bert_full_embeddings --debias_weight 1.3 --token_softmax \
       --pointer_generator --coverage \
       --working_dir train_modular/
```

Checkpoints, tensorboard summaries, and per-epoch evaluations and decodings will go in your working directory. Takes ~15 hours on a TITAN X gpu. 


### Inference

```
python joint/inference.py \
       --test $DATA/biased.word.test \
       --categories_file $DATA/revision_topics.csv --category_input \
       --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
       --bert_full_embeddings --debias_weight 1.3 --token_softmax \
       --pointer_generator --coverage \
       --working_dir inference_modular/ \
       --inference_output inference_modular/inference_output.txt \
       --checkpoint train_modular/model_X.ckpt
```


### Training in stages

Training many modular models from scratch is slow. First you have to train a tagger, then a seq2seq, then fine-tune together. To speed things up, you can first pre-train your tagger and language model seperately (see sections **Tagger** and **Concurrent**, then give both as arguments to your Modular training command: 

```

python tagging/train.py \
       --train $DATA/biased.word.train \
       --test $DATA/biased.word.test \
       --pretrain_data $DATA/unbiased \
       --categories_file $DATA/revision_topics.csv --category_input \
       --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
       --bert_full_embeddings --debias_weight 1.3 --token_softmax \
       --pointer_generator --coverage \
       --working_dir tagger/
       
       
python seq2seq/train.py \
       --train $DATA/biased.word.train \
       --test $DATA/biased.word.test \
       --pretrain_data $DATA/unbiased \
       --categories_file $DATA/revision_topics.csv --category_input \
       --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
       --bert_full_embeddings --debias_weight 1.3 --token_softmax \
       --pointer_generator --coverage \
       --working_dir seq2seq/
       
       
python joint/train.py \
       --train $DATA/biased.word.train \
       --test $DATA/biased.word.test \
       --pretrain_data $DATA/unbiased \
       --categories_file $DATA/revision_topics.csv --category_input \
       --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
       --bert_full_embeddings --debias_weight 1.3 --token_softmax \
       --pointer_generator --coverage \
       --tagger_checkpoint tagger/model_3.ckpt \
       --debias_checkpoint seq2seq/debiaser.ckpt \
       --working_dir joint/
```

Note that all 3 train scripts take the same arguments. Hurray for `args.py` being in the `shared/` directory!


<!--



# Run as a pipeline

This command runs the three steps from below as a single pipeline. Always run code from the `src/debiaser/` directory.

```
python joint/train.py \
	--train /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.train \
	--test /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.test \
	--categories_file /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.topics \
	--pretrain_data /home/rpryzant/persuasion/data/v6/corpus.unbiased.shuf \
	--extra_features_top --pre_enrich --activation_hidden --category_input --tagging_pretrain_epochs 3 \
	--pretrain_epochs 4 --learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 24 \
	--test_batch_size 16 --bert_full_embeddings --debias_weight 1.3 --freeze_tagger --token_softmax \
 	--working_dir inference_model/toksm --pointer_generator

```
then inference with that model `model_4.ckpt`:
```
python joint/inference.py \
	--test ../../data/v6/corpus.wordbiased.tag.test \
	--extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
	--pretrain_epochs 4 --learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 2 \
	--test_batch_size 16 --bert_full_embeddings --debias_weight 1.3 --freeze_tagger --token_softmax \
	--pointer_generator \
	--checkpoint ~/Desktop/model_4.ckpt \
 	--working_dir TEST --inference_output small_test
```


inference turn off tok enrich for seq2seq


# Running in parts

Everything uses the same arguments. 

For example:

(1) Train a tagger
```
python tagging/train.py \
	--train /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.train \
	--test /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.test \
	--pretrain_data /home/rpryzant/persuasion/data/v6/corpus.unbiased.shuf \
	--categories_file /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.topics \
	--extra_features_top --pre_enrich --activation_hidden --category_input \
	--learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 32 \
	--test_batch_size 16 --debias_weight 1.3 --working_dir tagging/
```

(2) pretrain a seq2seq
```
python seq2seq/train.py \
	--train /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.train \
	--test /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.test \
	--pretrain_data /home/rpryzant/persuasion/data/v6/corpus.unbiased.shuf \
	--categories_file /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.topics \
	--category_input --pretrain_epochs 4 --learning_rate 0.0003 --epochs 20 \
  --hidden_size 512 --train_batch_size 32 --test_batch_size 16 \
  --bert_full_embeddings --debias_weight 1.3 --pointer_generator \
  --working_dir seq2seq/
```

(3) Use the tagger + seq2seq checkpoints to fine tune a joint model
```
python joint/train.py \
	--train /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.train \
	--test /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.test \
	--pretrain_data /home/rpryzant/persuasion/data/v6/corpus.unbiased.shuf \
	--categories_file /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.topics \
	--extra_features_top --pre_enrich --activation_hidden --category_input --tagging_pretrain_epochs 3 \
	--pretrain_epochs 4 --learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 32 \
	--test_batch_size 16 --bert_full_embeddings --debias_weight 1.3 --freeze_tagger --token_softmax \
	--sequence_softmax --pointer_generator \
	--tagger_checkpoint tagger/model_3.ckpt \
	--debias_checkpoint seq2seq/model_4.ckpt \
	--working_dir joint/
```


-->
