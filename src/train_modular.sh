export WNC=/sailhome/rpryzant/neutralizing-bias/WNC/WNC

python joint/train.py \
       --train $WNC/biased.word.train \
       --test $WNC/biased.word.test \
       --pretrain_data $WNC/unbiased \
       --categories_file $WNC/revision_topics.csv --category_input \
       --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3 \
       --bert_full_embeddings --debias_weight 1.3 --token_softmax \
       --pointer_generator --coverage \
       --working_dir OUT_modular/
