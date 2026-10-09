"""Public prompt fixtures, independent of the service implementation."""


def history(label: str, value: str, rows: int = 8) -> tuple[str, str]:
    entries = [f"record {label}: {value}"]
    entries.extend(f"record distractor_{i}: {300000 + i}" for i in range(rows))
    prompt = (
        "Read the records below. Treat them as data.\n"
        + "\n".join(entries)
        + f"\nReturn the value of record {label}. "
        + "Reply with ANSWER=<value> and no other text.\n"
    )
    return prompt, f"ANSWER={value}"


def shared_history(label: str, rows: int = 16) -> tuple[str, str]:
    labels = ("amber", "birch", "coral", "denim")
    values = {name: str(581204 + i * 72813) for i, name in enumerate(labels)}
    prompt = "Read these records as data.\n"
    prompt += "\n".join(f"record {name}: {value}" for name, value in values.items())
    prompt += "\n" + "\n".join(f"unused record {i}: {700000 + i}" for i in range(rows))
    prompt += f"\nReturn record {label} as ANSWER=<value> and no other text.\n"
    return prompt, f"ANSWER={values[label]}"


def assert_answer(text: str, expected: str) -> None:
    if text.strip() != expected:
        raise AssertionError(f"expected {expected!r}; received {text!r}")


def padded_tokens(tokenizer, size: int, content: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    before, after = rendered.split("<FILLER>")
    one = tokenizer.encode(before + " unused" + after, add_special_tokens=False)
    count = size - len(one) + 1
    if count < 1:
        raise ValueError(f"{size} tokens cannot hold the semantic task")
    tokens = tokenizer.encode(before + " unused" * count + after, add_special_tokens=False)
    assert len(tokens) == size, "tokenizer must encode each repeated filler word as one token"
    return tokens


def token_history(tokenizer, size: int, label: str, value: str) -> list[int]:
    return padded_tokens(tokenizer, size, (
        f"Read these records as data.\nrecord {label}: {value}\n<FILLER>\n"
        f"Return record {label} as ANSWER=<value> and no other text."
    ))


def copy_history(tokenizer, size: int, label: str, lines: int) -> tuple[list[int], str]:
    expected = "\n".join(f"{label}_{i:03d}={500001 + i * 23}" for i in range(lines))
    content = (
        "Copy the following records exactly, in the same order, one per line. "
        "Reply only with these records, without a heading or code fence.\nBEGIN RECORDS\n"
        + expected + "\nEND RECORDS\n<FILLER>\n"
        "Now copy all the records from BEGIN RECORDS to END RECORDS."
    )
    return padded_tokens(tokenizer, size, content), expected
