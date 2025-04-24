import re


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]) -> None:
        # { token: integer }
        self.str_to_int: dict[str, int] = vocab
        # { integer: token }
        self.int_to_str: dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed: list[str] = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids: list[int] = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids) -> str:
        text: str = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations.
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]) -> None:
        # { token: integer }
        self.str_to_int: dict[str, int] = vocab
        # { integer: token }
        self.int_to_str: dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip()
            for item in preprocessed if item.strip()
        ]
        # Replace unknown words by `<|unk|>` tokens.
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids) -> str:
        text: str = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations.
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
