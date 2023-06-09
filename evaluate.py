from typing import Optional

import tensorflow as tf
import tensorflow_text as tft
import numpy as np
import config


class SentencepieceTokenizerLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(SentencepieceTokenizerLayer, self).__init__()
        model_prefix = f"{config.SPM_TRAINER_CONFIG['model_prefix']}.model"
        self.tokenizer = tft.SentencepieceTokenizer(  # note that fast version is also available
            model=open(model_prefix, "rb").read(),
            out_type=tf.int32,  # setting out_type as tf.string is also possible
            add_bos=True,
            add_eos=True,
            name="tokenizeText"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        tokenize text within inputs argument into sequence of tokens
        :param inputs: series of raw text in tf.Tensor type
        :return: square tensor whose element contains sequence of tokens
        """
        x = tf.strings.lower(
            input=inputs,
            encoding="utf-8",
            name="toLowerCase",
        )
        x = tf.strings.regex_replace(
            input=x,
            pattern="[^a-z0-9 ]",
            rewrite=" ",
            replace_global=True,
            name="normalizeText",
        )
        return (
            self.tokenizer.tokenize(x)  # ragged tensor
            .to_tensor(config.SPM_TRAINER_CONFIG["pad_id"])  # 0-filled square tensor
        )


def print_tokenize_result(
    text: str,
    tokenize_layer: SentencepieceTokenizerLayer,
    max_len: Optional[int] = None,
):
    result = tokenize_layer(tf.constant([text]))
    if max_len is None:
        result = result.numpy()
    else:
        result = postprocess_tensor(result, max_len)
    result = [
        (
            tokenize_layer.tokenizer
            .id_to_string(token_id)  # <tf.Tensor: shape=(), dtype=string, numpy=b'\xe2\x96\x81you'>
            .numpy()                 # b'\xe2\x96\x81you'
            .decode("utf-8")         # '_you'
        )
        for token_id in result[0]    # result is 2-dimensional since input text was wrapped in list
    ]
    print(f"{text} -> {result}")


def postprocess_tensor(x: tf.Tensor, max_len: int) -> np.array:
    """
    attach pad tokens to fill predefined max length
    :param x: sequence of tokens in tf.Tensor type
    :param max_len: predefined max length of each token sequences
    :return: pad-token filled numpy array
    """
    return tf.keras.utils.pad_sequences(
        sequences=x,
        maxlen=max_len,
        dtype="int32",
        padding="post",
        truncating="post",
        value=config.SPM_TRAINER_CONFIG["pad_id"]
    )
