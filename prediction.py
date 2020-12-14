from torch.autograd import Variable
import pickle
from preprocessing import cap_feature
import numpy as np
import torch


sentence_list = [
    "i think at this time we 're just in a a rebuilding type of phase you know",
    "well it just i- i- it 's so discouraging that they can n't really",
    "well that 's a that 's that 's good",
    "and so i i felt like number one you nee- you also need to to see what kind of name the like you mentioned before the name the college and university can can give you and another thing that uh another reason why i chose that was the the finances",
    "there 's an old saying you can l- you can lead a horse to water",
    "an- and now he 's probably an old age of uh twenty five or six and making all that money",
    "well that 's uh that 's quite a that 's a little more of a challenge for a young man than they they need at that point in their life i think"
]
input_sentence = input("Give your text to remove disfluencies (Or say 'options' to try test samples): ")
if input_sentence == "options":
    print("*"*150)
    for i in range(len(sentence_list)):
        print("[%i]: %s"% (i+1, sentence_list[i]))
    print("*"*150)
    option = int(input("Select one of the number: "))

    input_sentence = sentence_list[option-1]

model = torch.load("./models/BiLSTM.pt")

tokens = input_sentence.split(" ")

with open('./models/mapping.pkl', 'rb') as file:
    mapping = pickle.load(file)

word_ids = [mapping['word_to_id'][word] for word in tokens]
word_ids = Variable(torch.LongTensor(word_ids))

char_ids = [[mapping['char_to_id'][c] for c in word] for word in tokens]

chars_length = [len(c) for c in char_ids]
chars_length = torch.LongTensor(chars_length)

char_maxl = max(chars_length)

char_mask = np.zeros(
    (len(chars_length), char_maxl), dtype='int')
for i, c in enumerate(char_ids):
    char_mask[i, :chars_length[i]] = c
char_mask = Variable(torch.LongTensor(char_mask))

caps = [cap_feature(word) for word in tokens]
caps = Variable(torch.LongTensor(caps))

val, pred_tags_ids = model(word_ids.cuda(), char_mask.cuda(), caps.cuda(), chars_length.cuda(), {})

pred_tags = [mapping['id_to_tag'][tag] for tag in pred_tags_ids]

print("#"*150)
# print("Predictions: ", [(token, pred_tag) for token, pred_tag in zip(tokens, pred_tags)])
print("input:", input_sentence)
print("Output:", " ".join([token for token, pred_tag in zip(tokens, pred_tags) if pred_tag not in ["BE", "IE", "IP", "BE_IP", "C_IP", "C_IE"]]))
print("#"*150)


