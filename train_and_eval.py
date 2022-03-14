import numpy as np
from torch.autograd import Variable
import torch
import time
import matplotlib.pyplot as plt

def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    if type(tok) == torch.Tensor:
        tok = tok.item()

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """

    # We assume by default the tags lie outside a named entity
    default = tags["O"]

    idx_to_tag = {idx: tag for tag, idx in tags.items()}

    chunks = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                # Initialize chunk for each entity
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                # If chunk class is B, i.e., its a beginning of a new named entity
                # or, if the chunk type is different from the previous one, then we
                # start labelling it as a new entity
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def evaluating(model, datas, best_F, parameters, tag_to_id, dataset="Train"):
    '''
    The function takes as input the model, data and calcuates F-1 Score
    It performs conditional updates
     1) Flag to save the model
     2) Best F-1 score
    ,if the F-1 score calculated improves on the previous F-1 score
    '''
    # Initializations
    prediction = []  # A list that stores predicted tags
    save = False  # Flag that tells us if the model needs to be saved
    new_F = 0.0  # Variable to store the current F1-Score (may not be the best)
    correct_preds, total_correct, total_preds = 0., 0., 0.  # Count variables

    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if parameters['char_mode'] == 'CNN':
            d = {}

            # Padding the each word to max word size of that sentence
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))

        # We are getting the predicted output from our model
        if parameters['use_gpu']:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, chars2_length, d)
        predicted_id = out

        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks = set(get_chunks(ground_truth_id, tag_to_id))
        lab_pred_chunks = set(get_chunks(predicted_id, tag_to_id))

        # Updating the count variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    # Calculating the F1-Score
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    new_F = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print("{}: new_F: {} best_F: {} ".format(dataset, new_F, best_F))

    # If our current F1-Score is better than the previous best, we update the best
    # to current F1 and we set the flag to indicate that we need to checkpoint this model

    if new_F > best_F:
        best_F = new_F
        save = True

    return best_F, new_F, save


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_data, dev_data, test_data, tag_to_id, parameters, model_name):
    # Initializing the optimizer
    # The best results in the paper where achived using stochastic gradient descent (SGD)
    # learning rate=0.015 and momentum=0.9
    # decay_rate=0.05

    learning_rate = 0.015
    momentum = 0.9
    number_of_epochs = parameters['epochs']
    decay_rate = 0.05
    gradient_clip = parameters['gradient_clip']
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # variables which will used in training process
    losses = []  # list to store all losses
    loss = 0.0  # Loss Initializatoin
    best_dev_F = -1.0  # Current best F-1 Score on Dev Set
    best_test_F = -1.0  # Current best F-1 Score on Test Set
    best_train_F = -1.0  # Current best F-1 Score on Train Set
    all_F = [[0, 0, 0]]  # List storing all the F-1 Scores
    eval_every = len(train_data)  # Calculate F-1 Score after this many iterations
    plot_every = 2000  # Store loss after this many iterations
    count = 0  # Counts the number of iterations

    tr = time.time()
    model.train(True)
    for epoch in range(1, number_of_epochs):
        for i, index in enumerate(np.random.permutation(len(train_data))):
            count += 1
            data = train_data[index]

            ##gradient updates for each data entry
            model.zero_grad()

            sentence_in = data['words']
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']
            chars2 = data['chars']

            if parameters['char_mode'] == 'LSTM':
                chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
                d = {}
                for i, ci in enumerate(chars2):
                    for j, cj in enumerate(chars2_sorted):
                        if ci == cj and not j in d and not i in d.values():
                            d[j] = i
                            continue
                chars2_length = [len(c) for c in chars2_sorted]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
                for i, c in enumerate(chars2_sorted):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            if parameters['char_mode'] == 'CNN':

                d = {}

                ## Padding the each word to max word size of that sentence
                chars2_length = [len(c) for c in chars2]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                for i, c in enumerate(chars2):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            targets = torch.LongTensor(tags)

            # we calculate the negative log-likelihood for the predicted tags using the predefined function
            if parameters['use_gpu']:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(),
                                                              chars2_length, d)
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, chars2_length, d)
            loss += neg_log_likelihood.item() / len(data['words'])
            neg_log_likelihood.backward()

            # we use gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
            optimizer.step()

            # Storing loss
            if count % plot_every == 0:
                loss /= plot_every
                print(count, ': ', loss)
                if losses == []:
                    losses.append(loss)
                losses.append(loss)
                loss = 0.0

            # Evaluating on Train, Test, Dev Sets
            if count % (eval_every * 4) == 0 and count != 0:
                model.train(False)
                best_train_F, new_train_F, _ = evaluating(model, train_data, best_train_F, parameters, tag_to_id, "Train")
                best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F, parameters, tag_to_id, "Dev")
                if save:
                    print("Saving Model to ", model_name)
                    torch.save(model.state_dict(), model_name)
                best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F, parameters, tag_to_id, "Test")

                all_F.append([new_train_F, new_dev_F, new_test_F])
                model.train(True)

            # Performing decay on the learning rate
            if count % len(train_data) == 0:
                adjust_learning_rate(optimizer, lr=learning_rate / (1 + decay_rate * count / len(train_data)))

    model.train(False)
    best_train_F, new_train_F, _ = evaluating(model, train_data, best_train_F, parameters, tag_to_id, "Train")
    best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F, parameters, tag_to_id, "Dev")
    if save:
        print("New best dev_F! Saving Model to ", model_name)
        torch.save(model.state_dict(), model_name)
    best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F, parameters, tag_to_id, "Test")
    all_F.append([new_train_F, new_dev_F, new_test_F])

    time_taken = time.time() - tr
    print(time_taken)
    #plt.title('Training Losses')
    #plt.plot(losses, 'g', label='training loss')
    #plt.legend()
    #plt.show()

    return model, losses, all_F, time_taken