# BiLSTM_CRF_disfluency_detection

#### Model

To detect the disfluency and tagging of sequences, I have used BiLSTM-CRF model. This model can adequately use both past and future input features, thanks to BiLSTM. This model also can use sentence level tag information, thanks to CRF layer.

#### Dataset

Dataset used to train the disfluency detection is taken from [here](https://github.com/vickyzayats/switchboard_corrected_reannotated).

#### Tagging

BE - beginning of the reparandum

IE - inside the reparandum

IP - the last word before the interruption point

BE_IP - single token reparandum

C - repair (correction)

O - non-disfluency

C_IE - the word is both in the reparandum and repair but not before interruption point (in nested disfluencies)

C_IP - the word is both in the reparandum and repair and the last before the interruption point (in nested disfluencies)

#### word embeddings

Download the glove embedding zip file [glove.6B.zip](https://nlp.stanford.edu/projects/glove/) and extract *glove.6B.100d.txt* in the *models* directory.

#### Training:

To train the model run following command:

```
$ python train.py
```

#### Prediction:

Download the [trained model](https://drive.google.com/file/d/1SALwB2PrspK46R3aR6GLguXHoHObCJ8S/view?usp=sharing) and [mapping file](https://drive.google.com/file/d/1rcJRlmP7oyGjF4ytBhYidG_q9ob9t5ki/view?usp=sharing). Save this in the *models* directory.

To use the trained model for prediction run below command:

```
$ python prediction.py
```

#### Reference code

[NER-pytorch](https://github.com/ZhixiuYe/NER-pytorch)

#### Reference papers

[Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)

[Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
