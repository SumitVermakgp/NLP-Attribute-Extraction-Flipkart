# NLP Attribute Extraction 

This repo implements a NER model using Tensorflow ( Satcked Bi-directional LSTM + CRF + chars embeddings).


## Task

Given a sentence, give a tag to each word. 

```
Komnil  Men's  cerulean  blue     graphical  scooter  half     sleeve    round   neck    tshirt.
B_BRAND  O      O       B_COLOR     O         O      B_SLEEVE  I_SLEEVE   B_TYPE  I_TYPE  O
```


## Model

Similar to [Lample et al.](https://arxiv.org/abs/1603.01360) and [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a stacked bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF



## Getting started


1. Download the GloVe vectors with

```
make glove
```

Alternatively, you can download them manually [here](https://nlp.stanford.edu/projects/glove/) and update the `glove_filename` entry in `config.py`. You can also choose not to load pretrained word vectors by changing the entry `use_pretrained` to `False` in `model/config.py`.

2. Build the training data, train and evaluate the model with
```
make run
```


## Details


Here is the breakdown of the commands executed in `make run`:

1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```
python build_data.py
```

2. Train the model with

```
python train.py
```


3. Evaluate and interact with the model with
```
python evaluate.py
```


Data iterators and utils are in `model/data_utils.py` and the model with training/test procedures is in `model/ner_model.py`

Training time on NVidia Tesla K80 is 110 seconds per epoch on CoNLL train set using characters embeddings and CRF.



## Training Data


The training data must be in the following format.

A default test file is provided to help you getting started.


```
Komnil B_BRAND
Men'S O
Cerulean O
Blue B_COLOR
Graphical O
Scooter O
Half B_SLEEVE
Sleeve I_SLEEVE
Round B_TYPE
Neck I_TYPE
T-shirt O
. O
```


Once you have produced your data files, change the parameters in `config.py` like

```
# dataset
dev_filename = "data/coNLL/eng/eng.testa.iob"
test_filename = "data/coNLL/eng/eng.testb.iob"
train_filename = "data/coNLL/eng/eng.train.iob"
```

# Make the following folders:
Sequence tagging [build_data.py, evaluate.py , train.py, model,data]

model [base_model.py, config.py, data_utils.py, general_utils.py,ner_model.py]

data [xyz_test.txt , xyz_train.txt] // Put Glove pretrained vector [300d] in this folder.

