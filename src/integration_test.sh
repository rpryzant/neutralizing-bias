echo "STARTING tagging/train.py"
python tagging/train.py --train ../WNC/WNC/biased.word.train --test ../WNC/WNC/biased.word.test --debug_skip --hidden_size 16 --working_dir TEST --max_seq_len 20 --train_batch_size 2 --test_batch_size 2 --epochs 2
echo "STARTING seq2seq/train.py"
python seq2seq/train.py  --train ../WNC/WNC/biased.word.train --test ../WNC/WNC/biased.word.test --debug_skip --hidden_size 16 --working_dir TEST --max_seq_len 20 --train_batch_size 2 --test_batch_size 2 --epochs 2
echo "STARTING joint/train.py"
python joint/train.py   --train ../WNC/WNC/biased.word.train --test ../WNC/WNC/biased.word.test --debug_skip --hidden_size 16 --working_dir TEST --max_seq_len 20 --train_batch_size 2 --test_batch_size 2 --epochs 2