import argparse
import urllib

import torch

from data import *
from model import BiLSTM_CRF

from train_and_eval import train, demo

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CE7455 Assignment 2 - Named Entity Recognition')
    parser.add_argument('--train', type=str, default='./data/eng.train',
                        help='path to training data')
    parser.add_argument('--dev', type=str, default='./data/eng.testa',
                        help='path to validation data')
    parser.add_argument('--test', type=str, default='./data/eng.testb',
                        help='path to test data')
    parser.add_argument('--tag_scheme', type=str, default='BIOES', help='NER tagging scheme - either BIO or BIOES')
    parser.add_argument('--lower', action='store_true', help="Boolean variable to control lowercasing of words "
                                                             "(remove arg to disable)")
    parser.add_argument('--zeros', action='store_true', help="Boolean variable to control replacement of  "
                                                             "all digits by 0. (remove arg to disable)")
    parser.add_argument('--char_dim', type=int, default=30, help="Char embedding dimension")
    parser.add_argument('--word_dim', type=int, default=100, help="Token embedding dimension")
    parser.add_argument('--word_lstm_dim', type=int, default=200, help="Token LSTM hidden layer size")
    parser.add_argument('--word_bidirect', action='store_true', help="Use a bidirectional LSTM for words (remove arg "
                                                                     "for uni directional)")
    parser.add_argument('--embedding_path', default="./data/glove.6B.100d.txt",
                        help="Location of pretrained embeddings")
    parser.add_argument('--all_emb', action='store_true', help="Load all embeddings (remove arg to disable)")
    parser.add_argument('--crf', action='store_true', help="Use CRF (remove arg to disable)")
    parser.add_argument('--dropout', type=float, default=0.5, help="Droupout on the input (0 = no dropout)")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to run")
    parser.add_argument('--weights', type=str, default="", help="path to Pretrained for from a previous run")
    parser.add_argument('--name', type=str, default="self-trained-model", help="Model name")
    parser.add_argument('--gradient_clip', type=float, default=5.0)
    parser.add_argument('--char_mode', type=str, default="CNN", help="char encoder mode (LSTM or CNN)")
    parser.add_argument('--word_mode', type=str, default="CNN", help="word encoder mode (LSTM or CNN)")
    parser.add_argument('--num_word_cnn_layers', type=int, default=1, help="number of layers of cnn word encoder")
    parser.add_argument('--dilation', action='store_true', help="set to enable dilation in cnn word encoder")
    parser.add_argument('--use_gpu', action='store_true', help="remove arg if no CUDA GPU available")
    parser.add_argument("--reload", type=str, default="", help="path to pre-trained model")
    parser.add_argument('--load_from_mapping_file', action='store_true', help='load data from ./data/mapping.pkl')
    parser.add_argument('--eval_only', action='store_true', help="load model from ./models/self-trained-model "
                                                                 "and test it")
    parameters = vars(parser.parse_args())

    parameters['use_gpu'] = torch.cuda.is_available()  # GPU Check

    models_path = "./models/"  # path to saved models

    log_file = "./log.txt"
    # paths to files
    # To stored mapping file
    mapping_file = './data/mapping.pkl'

    # To stored model
    name = parameters['name']
    model_name = models_path + name  # get_name(parameters)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    train_sentences = load_sentences(parameters['train'], parameters['zeros'])
    test_sentences = load_sentences(parameters['test'], parameters['zeros'])
    dev_sentences = load_sentences(parameters['dev'], parameters['zeros'])

    train_data, dev_data, test_data, word_to_id, char_to_id, tag_to_id, word_embeds = \
        get_dataset(train_sentences, dev_sentences, test_sentences, parameters, mapping_file)




    print("{} / {} / {} sentences in train / dev / test.".format(len(train_data), len(dev_data), len(test_data)))

    # creating the model using the Class defined above
    model = BiLSTM_CRF(vocab_size=len(word_to_id),
                       tag_to_ix=tag_to_id,
                       embedding_dim=parameters['word_dim'],
                       hidden_dim=parameters['word_lstm_dim'],
                       use_gpu=parameters['use_gpu'],
                       char_to_ix=char_to_id,
                       pre_word_embeds=word_embeds,
                       use_crf=parameters['crf'],
                       char_mode=parameters['char_mode'],
                       word_mode=parameters['word_mode'],
                       num_word_cnn_layers=parameters['num_word_cnn_layers'],
                       dilation=parameters['dilation'])
    print("Model Initialized!!!")

    # Reload a saved model, if parameter["reload"] is set to a path
    if parameters['reload'] != '':
        if not os.path.exists(parameters['reload']):
            print("downloading pre-trained model")
            model_url = "https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/raw/master/trained-model-cpu"
            urllib.request.urlretrieve(model_url, parameters['reload'])

        model.load_state_dict(torch.load(parameters['reload']))
        print("model reloaded :", parameters['reload'])

    if parameters['use_gpu']:
        model.cuda()

    if not parameters['eval_only']:
        _, losses, all_F, time_taken = train(model, train_data, dev_data, test_data, tag_to_id, parameters, model_name)

        with open(log_file,'a') as lf:
            lf.write('*'*100+"\n")
            lf.write('Char mode:: {}, word mode:: {}, CNN Layers:: {}, Dilation:: {}, CRF:: {} \n'.
                     format(parameters['char_mode'], parameters['word_mode'], parameters['num_word_cnn_layers'],
                            parameters['dilation'], parameters['crf']))
            lf.write('Model parameters:: '+str(count_parameters(model))+'\n')
            lf.write('Losses:: '+str(losses)+'\n')
            lf.write('all_F:: '+str(all_F)+'\n')
            lf.write('Time taken:: {}\n'.format(time_taken))
            lf.write('*' * 100 + "\n")

        model.load_state_dict(torch.load(model_name))




    else:
        model.load_state_dict(torch.load(model_name))
        model_testing_sentences = ['Ron went to the USA', 'Germany defeated Argentina in the 2014 FIFA World Cup final']

        demo(model, parameters, model_testing_sentences, word_to_id, char_to_id, tag_to_id)






