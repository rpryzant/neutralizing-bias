
## Usage

**Train**: `python train.py --config sample_config.json --bleu`

* logs are written into `[working dir]/train_log`
* tb summaries go into `[working dir]/graphs.csv`
* models are saved on each epoch, into `[working_dir]/model.[epoch]`

**Inference**: `python inference.py --config sample_config.json --models model_path1,model_path2,...`

## Configs

See `sample_config.json` for an example.