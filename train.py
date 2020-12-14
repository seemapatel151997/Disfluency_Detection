from preprocessing import load_sentences, word_mapping, augment_with_pretrained, char_mapping, tag_mapping, prepare_dataset
import optparse
from collections import OrderedDict
import itertools
import os
import numpy as np
import codecs
import torch
import pickle as cPickle
# import sys
from model import BiLSTM_CRF
from torch.autograd import Variable
from utils import eval_temp, eval_script, adjust_learning_rate
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


data_path = "./data/"
mapping_file = 'models/mapping.pkl'
models_path = "models/"

optparser = optparse.OptionParser()

optparser.add_option(
    "-l", "--lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-p", "--pre_emb", default="models/glove.6B.100d.txt",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="1",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="200",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    '--name', default='BiLSTM',
    help='model name'
)
optparser.add_option(
    '--epoches', default=20,
    help='number of epoches for training'
)
optparser.add_option(
    '--lr', default=0.015,
    help='learning rate'
)
optparser.add_option(
    '--momentum', default=0.9,
    help='momentum for optimizer'
)

opts = optparser.parse_args()[0]

parameters = OrderedDict()
parameters['lower'] = opts.lower == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['crf'] = opts.crf == 1
parameters['char_mode'] = opts.char_mode
parameters['reload'] = opts.reload == 1
parameters['name'] = opts.name
parameters['epoches'] = opts.epoches
parameters['lr'] = opts.lr
parameters['momentum'] = opts.momentum

parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()
use_gpu = parameters['use_gpu']

lower = parameters['lower']

name = parameters['name']
model_name = models_path + name + ".pt"  # get_name(parameters)

train_sentences = load_sentences(data_path, lower, "train")
dev_sentences = load_sentences(data_path, lower, "dev")
test_train_sentences = load_sentences(data_path, lower, "train_test")

dico_words_train = word_mapping(train_sentences, lower)[0]

dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences])
        ) if not parameters['all_emb'] else None
    )

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(data_path, "train")

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, data_path, "train", lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, data_path, "dev", lower
)
test_train_data = prepare_dataset(
    test_train_sentences, word_to_id, char_to_id, tag_to_id, data_path, "train_test", lower
)


print("%i / %i sentences in train / dev." % (
    len(train_data), len(dev_data)))

all_word_embeds = {}
for i, line in enumerate(codecs.open(opts.pre_emb, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

word_embeds = np.random.uniform(-np.sqrt(0.06),
                                np.sqrt(0.06), (len(word_to_id), opts.word_dim))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    cPickle.dump(mappings, f)

model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'],
                   char_embedding_dim=32)
                   # n_cap=4,
                   # cap_embedding_dim=10)

if parameters['reload']:
    model.load_state_dict(torch.load(model_name).state_dict())
if use_gpu:
    model.cuda()
learning_rate = parameters['lr']

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=parameters['momentum'])
losses = []
loss = 0.0
best_dev_F = -1.0
best_train_F = -1.0
plot_every = 100
eval_every = 5000
count = 0

def evaluating(model, datas, best_F1_score):
    prediction = []
    save = False
    F1_score = 0.0

    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    ground_truth_ids = []
    predicted_ids = []
    for data in datas:
        ground_truth_id = data['tags']
        ground_truth_ids.extend(ground_truth_id)
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

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
            chars2_mask = np.zeros(
                (len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros(
                (len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(),
                             dcaps.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        predicted_ids.extend(predicted_id)
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            # print(word, true_id, pred_id)
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    F1_score = f1_score(ground_truth_ids, predicted_ids, average="weighted")
    print("F1 Score: ", F1_score)
    if F1_score > best_F1_score:
        best_F1_score = F1_score
        save = True
    print('the best F1_score is ', best_F1_score)
    predf = eval_temp + '/pred.' + name
    scoref = eval_temp + '/score.' + name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))
    return best_F1_score, F1_score, save


model.train(True)
for epoch in range(1, int(parameters['epoches'])):
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        data = train_data[index]
        model.zero_grad()

        sentence_in = data['words']
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data['tags']
        chars2 = data['chars']

        # char lstm
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
            chars2_mask = np.zeros(
                (len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        # ######## char cnn
        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros(
                (len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        targets = torch.LongTensor(tags)
        caps = Variable(torch.LongTensor(data['caps']))
        if use_gpu:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(
            ), targets.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d)
        else:
            neg_log_likelihood = model.neg_log_likelihood(
                sentence_in, targets, chars2_mask, caps, chars2_length, d)
        loss += neg_log_likelihood.data / len(data['words'])
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        if count % plot_every == 0:
            loss /= plot_every
            print(count, ': ', loss)
            loss = 0.0

        if count % (eval_every) == 0 and count > (eval_every * 20) or \
                count % (eval_every*4) == 0 and count < (eval_every * 20):
            print("EPOCH: ", epoch)
            model.train(False)
            best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F)
            best_dev_F, new_dev_F, save = evaluating(
                model, dev_data, best_dev_F)
            if save:
                torch.save(model, model_name)
            model.train(True)
            
        if count % len(train_data) == 0:
            print("Learning_rate:", learning_rate/(1+0.05*count/len(train_data)))
            adjust_learning_rate(
                optimizer, lr = learning_rate/(1+0.05*count/len(train_data)))

plt.plot(losses)
plt.show()

print("Done!")