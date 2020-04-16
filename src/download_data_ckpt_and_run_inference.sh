wget http://nlp.stanford.edu/projects/bias/bias_data.zip
unzip bias_data.zip

wget https://nlp.stanford.edu/projects/bias/model.ckpt

python joint/inference.py \
       --extra_features_top --pre_enrich --activation_hidden \
       --test_batch_size 1 --bert_full_embeddings --debias_weight 1.3 --token_softmax \
       --pointer_generator --coverage \
       --working_dir TEST \
       --test bias_data/WNC/biased.word.test \
       --checkpoint model.ckpt \
       --inference_output TEST/output.txt