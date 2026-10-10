"""Independent answers whose source records span the full request history."""


def distributed_history(tokenizer, size, label, value_base, count):
    records = [f"{label}_{i:03d}={value_base + i * 23}" for i in range(count)]
    return positioned_records(tokenizer, size, records)


def positioned_records(tokenizer, size, records):
    count = len(records)
    cuts = (0, count // 3, count * 2 // 3, count)
    groups = ["\n".join(records[cuts[i]:cuts[i + 1]]) for i in range(3)]
    content = (
        "Copy every record from the three sections exactly, in order, one per line. "
        "Reply only with the records, without a heading or code fence.\nEARLY RECORDS\n"
        + groups[0] + "\n<PAD_A>\nMIDDLE RECORDS\n" + groups[1]
        + "\n<PAD_B>\nTAIL RECORDS\n" + groups[2]
        + "\nNow copy all records from EARLY, MIDDLE and TAIL, in that order."
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    base = rendered.replace("<PAD_A>", " unused").replace("<PAD_B>", " unused")
    extra = size - len(encode(base))
    first_extra = size // 2 - len(encode(rendered.split("<PAD_A>")[0] + " unused"))
    assert 0 <= first_extra <= extra, "input length cannot hold the three source sections"
    rendered = rendered.replace("<PAD_A>", " unused" * (first_extra + 1))
    rendered = rendered.replace("<PAD_B>", " unused" * (extra - first_extra + 1))
    tokens = encode(rendered)
    assert len(tokens) == size
    spans = []
    for name, group in zip(("early", "middle", "tail"), groups):
        start = rendered.index(group)
        spans.append({"section": name, "start_token": len(encode(rendered[:start])),
                      "end_token": len(encode(rendered[:start + len(group)])),
                      "first_record": group.splitlines()[0], "last_record": group.splitlines()[-1]})
    return tokens, "\n".join(records), spans
