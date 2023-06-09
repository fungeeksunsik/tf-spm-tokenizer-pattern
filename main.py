import logging
import sys
import pathlib
import config
import preprocess
import train
import evaluate

from typer import Typer


app = Typer()
local_dir = pathlib.Path(config.LOCAL_DIR)
local_dir.mkdir(exist_ok=True, parents=True)

formatter = logging.Formatter(
    fmt="%(asctime)s (%(funcName)s) : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


@app.command(
    "preprocess",
    help="Process for downloading archived IMDb data and preprocessing it"
)
def run_preprocess():
    logger.info("Download IMDb dataset and unpack archive")
    imdb_path = preprocess.download_and_unpack_imdb(
        source_url=config.SOURCE_URL,
        local_dir=local_dir,
        archive_name=config.ARCHIVE_NAME,
    )

    logger.info("Extract review data from unpacked archive")
    imdb_data = preprocess.extract_data_from_imdb(imdb_path)

    logger.info("Split dataset into train, test and save each respectively")
    preprocess.split_save_imdb_data(imdb_data, local_dir)


@app.command(
    "train",
    help="Process for training tokenizer using normalized corpus of training data"
)
def run_train():
    logger.info("Extract reviews from train data and save into local directory")
    train.extract_corpus(local_dir)

    logger.info("Train tokenizer using extracted corpus")
    train.train_and_save_tokenizer(config.SPM_TRAINER_CONFIG)


@app.command(
    "evaluate",
    help="Process for evaluating Keras layer that utilizes trained tokenizer"
)
def run_evaluate():
    logger.info("Load sentencepiece tokenizer as Tensorflow layer")
    tokenize_layer = evaluate.SentencepieceTokenizerLayer()

    dummy_texts = [
        "You'll see that CAPITAL letters in the review",
        "is INDEED converted into lowercase letters,",
        "texts are normalized and eos, bos tokens are added."
    ]
    logger.info("Apply text tokenization on dummy texts")
    for text in dummy_texts:
        evaluate.print_tokenize_result(text, tokenize_layer)

    logger.info("Apply text tokenization with padding")
    for text in dummy_texts:
        evaluate.print_tokenize_result(text, tokenize_layer, max_len=20)


if __name__ == "__main__":
    app()
