
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
	--pretrain_data /home/rpryzant/persuasion/data/v6/corpus.unbiased.shuf \
	--extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
	--pretrain_epochs 4 \
	--learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 32 --test_batch_size 16 \
	--bert_full_embeddings --debias_weight 1.3 --freeze_tagger --token_softmax \
	--working_dir paper_runs/refactor_test1
```

# Running in parts

Everything uses the same arguments. 

For example:

(1) Train a tagger
```
python tagger/train.py
	--train /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.train \
	--test /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.test \
	--extra_features_top --pre_enrich --activation_hidden 
  --epochs 3 \
  --learning_rate 3e-5 \
  --working_dir tagger/
```

(2) pretrain a seq2seq
```
python seq2seq/train.py
	--train /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.train \
	--test /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.test \
	--pretrain_data /home/rpryzant/persuasion/data/v6/corpus.unbiased.shuf \
	--pretrain_epochs 4 \
  --epochs 0 \    # turn off regular training after pretraining
	--learning_rate 0.0003 \
  --hidden_size 512 \
  --train_batch_size 32 --test_batch_size 32
	--bert_full_embeddings
  --working_dir pretrain_seq2seq/
```

(3) Use the tagger + seq2seq checkpoints to fine tune a joint model
```
python joint/train.py
	--train /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.train \
	--test /home/rpryzant/persuasion/data/v6/corpus.wordbiased.tag.test \
	--pretrain_data /home/rpryzant/persuasion/data/v6/corpus.unbiased.shuf \
	--extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
	--pretrain_epochs 4 \
	--learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 32 --test_batch_size 16 \
	--bert_full_embeddings --debias_weight 1.3 --freeze_tagger --token_softmax \
  --tagger_checkpoint tagger/model_3.ckpt \
  --debias_checkpoint seq2seq/model_4.ckpt \
	--working_dir joint/
```


