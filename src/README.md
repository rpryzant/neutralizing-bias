
# Overview

This directory has the following submodules:
* `lexicons/`: text files with expert features from the literature (basically word lists).
* `tagging/`: The tagging model 
* `seq2seq/`: The seq2seq debiasing model 
* `joint/`: The joint model, which has a tagger and seq2seq inside of it
* `shared/`: Code that is shared between all submodules: data iterators, CLI arguments, constants

Each model directory (`tagging`, `seq2seq`, `joint`) has the following files:
* `model.py`: modeling code (whether that be the tagger, seq2seq, or joint model)
* `utils.py`: code for (1) training and (2) evaluation 
* `train.py`: main driver code that builds and trains a model, evaluating after each epoch




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


