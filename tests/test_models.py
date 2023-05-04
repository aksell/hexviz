from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from hexviz.models import get_zymctrl


def test_get_zymctrl():
    result = get_zymctrl()

    assert result is not None
    assert isinstance(result, tuple)

    tokenizer, model = result

    assert isinstance(tokenizer, GPT2TokenizerFast)
    assert isinstance(model, GPT2LMHeadModel)
