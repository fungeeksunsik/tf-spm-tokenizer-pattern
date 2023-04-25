# spm-tokenizer-pattern

This repository implements common pattern appears when training sentencepiece tokenizer from raw corpus and use it within Tensorflow framework. This repository basically replicates process implemented in [this repository](https://github.com/fungeeksunsik/hf-tokenizer-pattern). Namely, role of two main components contained in this repository can be described as:

* `preprocess.py` : fetch data from source, extract information from it and save train, test data
* `train.py` : extract corpus from train data, train and save tokenizer

Unlike huggingface tokenizer, sentencepiece tokenizer can be directly incorporated into Tensorflow graph(i.e. model). By doing so, corresponding NLP model can solely be saved and defined by Tensorflow SavedModel, which makes system that leverages the model relatively compact and simple. That being said, this repository also implements additional pattern within separate module explained below.

* `evaluate.py` : load trained spm tokenizer using tensorflow-text API and make use of it

These codes are implemented and executed on macOS environment with Python 3.9 version.

## Execute

To execute implemented process, change directory into the cloned repository and execute following:

```shell
python3 --version  # Python 3.9.5
python3 -m venv venv  # generate virtual environment for this project
source venv/bin/activate  # activate generated virtual environment
pip install -r requirements.txt  # install required packages
```

Then, implemented process can be directly executed by passing appropriate argument to main module. For example, to fetch IMDb data from web source and compose corpus, execute:

```shell
python3 main.py preprocess
```

Or, to train sentencepiece tokenizer from preprocessed corpus, execute:

```shell
python3 main.py train
```

Without any modification on configuration described in `config.py`, these processes will save trained sentencepiece tokenizer into local directory whose absolute path is: 

* `/tmp/spmTokenizer/tokenizer.model`

Given this result file, logics implemented in `evaluate.py` module can be executed with following command:

```shell
python3 main.py evaluate
```

`SentencepieceTokenizerLayer` class defined in `evaluate.py` implements text preprocessing logics as a series of Tensorflow operations. After loading trained tokenizer saved in the local directory when initialized, it applies following operations in order. Check [source code](hthttps://github.com/fungeeksunsik/spm-tokenizer-pattern/blob/master/evaluate.py#L22) for more details.

1. normalize strings
   1. `tf.strings.lower`: takes raw text tensor as input, converts upper case of every string within the tensor, returns processed tensor as output
   2. `tf.strings.regex_replace`: takes raw text tensor as input, converts every matching regular expression pattern exists in every string within the tensor, returns processed tensor as output 
2. tokenize normalized strings
   1. `SentencepieceTokenizer.tokenize`: takes tensor of strings as input and returns corresponding ragged tensor of tokens in user-defined data type
   2. `to_tensor`: method of ragged tensor that converts itself as square tensor. It makes length of arrays in ragged tensor even by filling input value(mostly, pad token)to reach maximum array length within input tensor

Mostly, dimension of token sequence has to be fixed when used as input of language model. As dimension of output tensor returned from `to_tensor` method can vary according to input batches, utility function to fill sequence into predefined constant maximum length is required. For this, Tensorflow provides [pad_sequence](https://github.com/fungeeksunsik/spm-tokenizer-pattern/blob/master/evaluate.py#L75) method which takes **iterable of iterables** as input and returns numpy array of sequences whose dimension is fixed as maximum length. This means that attaching `to_tensor` at the end of tokenization was essential, since ragged tensor itself is not iterable. To check, execute:

```python
import tensorflow as tf

hasattr(tf.ragged.constant([[0], [1,2], [3,4,5]]), "__iter__")  # False
```

To show how tokenizer actually works, main module prints the tokenization results on list of dummy texts. It is defined as

```python
dummy_texts = [
    "You'll see that CAPITAL letters in the review",
    "is INDEED converted into lowercase letters,",
    "texts are normalized and eos, bos tokens are added."
]
```

which is tokenized respectively into:

```python
[
    ['[CLS]', '▁you', '▁ll', '▁see', '▁that', '▁capital', '▁letters', '▁in', '▁the', '▁review', '[SEP]'],
    ['[CLS]', '▁is', '▁indeed', '▁converted', '▁into', '▁lower', 'case', '▁letters', '[SEP]'],
    ['[CLS]', '▁text', 's', '▁are', '▁normal', 'ized', '▁and', '▁e', 'os', '▁bos', '▁to', 'k', 'ens', '▁are', '▁added', '[SEP]']
]
```

and when sequence padding is applied:

```python
[
    ['[CLS]', '▁you', '▁ll', '▁see', '▁that', '▁capital', '▁letters', '▁in', '▁the', '▁review', '[SEP]', '[PAD]', ...],
    ['[CLS]', '▁is', '▁indeed', '▁converted', '▁into', '▁lower', 'case', '▁letters', '[SEP]', '[PAD]', '[PAD]', ...],
    ['[CLS]', '▁text', 's', '▁are', '▁normal', 'ized', '▁and', '▁e', 'os', '▁bos', '▁to', 'k', 'ens', '▁are', '▁added', '[SEP]', '[PAD]', '[PAD]', ...]
]
```