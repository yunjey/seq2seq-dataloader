# DataLoader for Seq2seq
This repository provides an usage example of PyTorch official data loader for text dataset. 

<br>


## Prerequesites
* [PyThon 2.7 or 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.1.12](http://pytorch.org/)
* [NLTK](http://www.nltk.org/)


<br>

## Usage 

#### 1. Clone the repository
```bash
$ git clone https://github.com/yunjey/seq2seq-dataloader.git
$ cd seq2seq-dataloader
```

#### 2. Download nltk tokenizer
```bash
$ pip install nltk
$ python
$ import nltk
$ nltk.download('punkt')
```

#### 3. Build word2id dictionary 

```bash
$ python build_vocab.py
```

#### 4. Check DataLoader
For usage, please see [example.ipynb](https://github.com/yunjey/seq2seq-dataloader/blob/master/example.ipynb).

