# Neutralizing Biased Text

We provide code and documentation to accompany our work "Automatically Neutralizing Biased Text", to be presented at AAAI-20. This library provides models to both detect biased language in Wikipedia articles, and generative models to generate 'de-biased' versions of biased sentences.

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
* `src/debiaser/`: The 
