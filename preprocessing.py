import spacy
import os
import codecs
import model
from utils import create_mapping, create_dico


nlp = spacy.load("en_core_web_sm")

def read_data(dir_path, data_type):
    '''
    Read data from the data files and giving as a list of sentences and list of labels.
    '''
    text_file_name = data_type+".txt"
    label_file_name = data_type+"_labels.txt"

    with open(dir_path+text_file_name, "r") as file1:
        sentences_lines = [line for line in file1.readlines()]
    with open(dir_path+label_file_name, "r") as file2:
        labels_lines = [l_line for l_line in file2.readlines()]
    return sentences_lines, labels_lines

def tokenize(text_list, lower):
    tokens_list = []
    for line in text_list:
        if lower:
            line = line.lower()
        tokens_list.append([[str(i)] for i in line[:-2].split(" ")])
    return tokens_list


def load_sentences(data_path, lower, data_type):
    '''
    Read and convert data into ....
    '''
    X_sentences, y_sentences = read_data(data_path, data_type)
    X_tokens = tokenize(X_sentences, lower)

    return X_tokens


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)

    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=3}
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<PAD>'] = 10000000
    # dico[';'] = 0
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char

def tag_mapping(data_path, data_type):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    with open(data_path+data_type+"_labels.txt", "r") as file1:
        tags = [line.split(" ")[:-1] for line in file1.readlines()]
    dico = create_dico(tags)
    dico[model.START_TAG] = -1
    dico[model.STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, data_path, data_type, lower=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []

    with open(data_path+data_type+"_labels.txt", "r") as file1:
        tags = [line.split(" ")[:-1] for line in file1.readlines()]
    
    for s, t in zip(sentences, tags):
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w] for w in t]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
        })
    return data