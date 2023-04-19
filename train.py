import pathlib
import pandas as pd
import sentencepiece as spm
from typing import Dict, Union


def extract_corpus(local_dir: pathlib.Path):
    """
    Extract corpus from train data with text normalization
    :param local_dir: local directory to save extracted corpus
    :return: save extracted corpus into local_dir and return nothing
    """
    train_data_path = local_dir.joinpath("train.csv")
    corpus = (
        pd.read_csv(train_data_path)["review"]
        .str.lower()
        .str.replace("[^a-z0-9 ]", " ", regex=True)
    )
    corpus_file_path = str(local_dir.joinpath("corpus.txt"))
    with open(corpus_file_path, "w") as file:
        file.writelines("\n".join(corpus))


def train_and_save_tokenizer(spm_trainer_config: Dict[str, Union[int, str, bool]]):
    """
    For more configuration options, see: https://github.com/google/sentencepiece/blob/master/doc/options.md
    :param spm_trainer_config: dictionary that maps configuration parameter to its value
    :return: train and save sentencepiece tokenizer and return nothing
    """
    spm.SentencePieceTrainer.Train(**spm_trainer_config)
