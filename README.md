#CE7455 Assignment 2 - Named Entity Recognition


```commandline
usage: main.py [-h] [--train TRAIN] [--dev DEV] [--test TEST]
               [--tag_scheme TAG_SCHEME] [--lower] [--zeros]
               [--char_dim CHAR_DIM] [--word_dim WORD_DIM]
               [--word_lstm_dim WORD_LSTM_DIM] [--word_bidirect]
               [--embedding_path EMBEDDING_PATH] [--all_emb] [--crf]
               [--dropout DROPOUT] [--epochs EPOCHS] [--weights WEIGHTS]
               [--name NAME] [--gradient_clip GRADIENT_CLIP]
               [--char_mode CHAR_MODE] [--word_mode WORD_MODE]
               [--num_word_cnn_layers NUM_WORD_CNN_LAYERS] [--dilation]
               [--use_gpu] [--reload RELOAD] [--load_from_mapping_file]
               [--eval_only]
```


```commandline
optional arguments:
  -h, --help                                    show this help message and exit
  --train TRAIN                                 path to training data
  --dev DEV                                     path to validation data
  --test TEST                                   path to test data
  --tag_scheme TAG_SCHEME                       NER tagging scheme - either BIO or BIOES
  --lower                                       arg to control lowercasing of words (remove arg to disable)
  --zeros                                       arg to control replacement of all digits by 0. (remove arg to disable)
  --char_dim CHAR_DIM                           Char embedding dimension
  --word_dim WORD_DIM                           Token embedding dimension
  --word_lstm_dim WORD_LSTM_DIM                 Token LSTM hidden layer size
  --word_bidirect                               Use a bidirectional LSTM for words (remove arg for uni directional)
  --embedding_path EMBEDDING_PATH               Location of pretrained embeddings
  --all_emb                                     Load all embeddings (remove arg to disable)
  --crf                                         Use CRF (remove arg to disable)
  --dropout DROPOUT                             Droupout on the input (0 = no dropout)
  --epochs EPOCHS                               Number of epochs to run
  --weights WEIGHTS                             path to Pretrained for from a previous run
  --name NAME                                   Model name
  --gradient_clip GRADIENT_CLIP     
  --char_mode CHAR_MODE                         char encoder mode (LSTM or CNN)
  --word_mode WORD_MODE                         word encoder mode (LSTM or CNN)
  --num_word_cnn_layers NUM_WORD_CNN_LAYERS     number of layers of cnn word encoder
  --dilation                                    set to enable dilation in cnn word encoder
  --use_gpu                                     remove arg if no CUDA GPU available
  --reload RELOAD                               path to pre-trained model
  --load_from_mapping_file                      load data from ./data/mapping.pkl
  --eval_only                                   load model from ./models/<model_name> and test it
```

For example, to train a model with a CNN char-level encoder and a 3-layer dilated CNN word-level encoder with CRF,
```commandline
python main.py --lower --zeros --word_bidirect --all_emb --crf --char_mode="CNN" --word_mode="CNN" --num_word_cnn_layers=3 --dilation --use_gpu --name='CNN_CNN3_DILATED'
```
To use the model trained by the above command for a demo on a set of example sentences,
```commandline
python main.py --load_from_mapping_file --lower --zeros --word_bidirect --all_emb --crf --char_mode="CNN" --word_mode="CNN" --num_word_cnn_layers=3 --dilation --use_gpu --name='CNN_CNN3_DILATED' --eval_only
```

#Link to repo
```
https://github.com/chandTavirs/NER
```
