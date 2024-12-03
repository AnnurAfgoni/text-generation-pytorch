def get_tokens(text):
    with open(text, "r") as f:
        text = f.read()
    return tuple(set(text))