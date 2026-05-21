import pytest

from backend.agents.rlm.primitives import _extract_json


def test_extract_json_bare_object():
    assert _extract_json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_extract_json_inside_code_fence():
    text = 'Here is the result:\n```json\n{"a": 1}\n```\nDone.'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_wrapped_in_prose():
    assert _extract_json('The plan is {"a": 1} and nothing else.') == {"a": 1}


def test_extract_json_tolerates_braces_inside_strings():
    assert _extract_json('{"note": "a closing } brace in text"}') == {
        "note": "a closing } brace in text"}


def test_extract_json_raises_when_no_json():
    with pytest.raises(ValueError):
        _extract_json("there is no json object here at all")
