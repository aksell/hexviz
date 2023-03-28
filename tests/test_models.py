
from transformers import (GPT2LMHeadModel, GPT2TokenizerFast, T5EncoderModel,
                          T5Tokenizer)

from hexviz.models import get_protT5, get_zymctrl


def test_get_protT5():
    result = get_protT5()

    assert result is not None
    assert isinstance(result, tuple)

    tokenizer, model = result

    assert isinstance(tokenizer, T5Tokenizer)
    assert isinstance(model, T5EncoderModel)

def test_get_zymctrl():
    result = get_zymctrl()

    assert result is not None
    assert isinstance(result, tuple)

    tokenizer, model = result

    assert isinstance(tokenizer, GPT2TokenizerFast)
    assert isinstance(model, GPT2LMHeadModel)