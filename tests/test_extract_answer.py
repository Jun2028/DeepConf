from src.extract_answer import extract_answer


def test_boxed_answer_basic() -> None:
    assert extract_answer(r"Here is it: \\boxed{123}") == 123


def test_boxed_answer_with_padding() -> None:
    assert extract_answer(r"\\boxed{ 007 }") == 7


def test_boxed_answer_with_extra_text() -> None:
    text = r"final is \\boxed{answer is 42 and done}"
    assert extract_answer(text) == 42


def test_fallback_last_integer() -> None:
    assert extract_answer("therefore 456") == 456


def test_boxed_takes_priority_over_fallback() -> None:
    text = r"\\boxed{12} but later says 999"
    assert extract_answer(text) == 12


def test_no_integer_returns_none() -> None:
    assert extract_answer("no digits here") is None


def test_out_of_range_returns_none() -> None:
    assert extract_answer(r"\\boxed{1000}") is None
    assert extract_answer("-1") is None


def test_whitespace_and_newlines() -> None:
    text = "result:\n\\boxed{\n 88 \n}\n"
    assert extract_answer(text) == 88