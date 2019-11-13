# Neutralizing Biased Text

We provide code and documentation to accompany our work "Automatically Neutralizing Biased Text", to be presented at AAAI-20. This library provides discriminative models to detect biased language in Wikipedia articles, and generative models to generate 'de-biased' versions of biased sentences.

## Requirements 

```
pip install torch torchvision
pip install pytorch-pretrained-bert
pip install tensorboardX
pip install simplediff
pip install nltk
pip install sklearn
```

## Installation

```bash
pip install -r requirements.txt
```

## Overview 
Our code-based is structured in the following format: 

* `harvest/`: Provides utilities for crawling Wikipedia articles and for generating a parallel dataset of biased-debiased sentences. Our data generation approach mirrors that proposed by Recasens et al. (https://nlp.stanford.edu/pubs/neutrality.pdf). A final version of our crawled dataset can be found at https://stanford.io/2Q8G3bX. The zip file containing the data is 100MB
and expands to 500MB. 
* `src/`: This folder provides the model architectures, and training procedures for both detecting bias and generating 'debiased' versions of text. It is sub-divided in the following manner: 
    + `src/tagging/`: Functionality for detecting bias in a given input sentence. The model architectures, which are based on BERT and use the huggingface implementation, can be found under model.py. Simple baselines we implement, such as logistic regression classifiers, are present in baseline.py. The primary training loop can be found under train.py. To spawn a basic run,
    you can call the following: 

    python tagging/train.py --train <training dataset> --test <test dataset> --working_dir <dir> --train_batch_size <batch_size> --test_batch_size <batch_size>  --hidden_size <hidden_size> --debug_skip

    + `src/seq2seq/`: 
    + `src/joint/`: 
    + `src/lexicons/`:
    + `src/shared/`: A set of utilities that are shared by both the bias detection and debias generation modules, such as an implementation of beam search. We also store, constants and arguments that are shared globally.
