import re

import tiktoken

from classes.text_tokenizer import SimpleTokenizerV1, SimpleTokenizerV2
from data.the_verdict import get_the_verdict

raw_text = get_the_verdict()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])

# Split text by regular expression.
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# Remove entries containing only whitespaces.
preprocessed = [item for item in preprocessed if item.strip()]
print("Tokens:", len(preprocessed))
print("Excerpt from tokens:", preprocessed[:30])

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
print("Vocabulary size (unique tokens):", len(all_tokens))

# Build vocabulary dict with `key: value` => `token: integer`.
print("Last vocabulary items:")
vocab = {token: integer for integer, token in enumerate(all_tokens)}
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

tokenizer = SimpleTokenizerV1(vocab)
text = """
"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride.
"""
ids = tokenizer.encode(text)
print("Token IDs (encode):", ids)
print("Words (decode):", tokenizer.decode(ids))

# This text will fail with V1:
# KeyError: 'Hello'
# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print("Example text:", text)

# Example usage for the `<|endoftext|>` and ´<|unk|>` token.
tokenizer = SimpleTokenizerV2(vocab)
print("Encoded and decoded text:", tokenizer.decode(tokenizer.encode(text)))

# Instantiate the `byte pair encoding` (BPE).
tokenizer = tiktoken.get_encoding("gpt2")
# tokenizer = tiktoken.encoding_for_model("gpt2")
# tokenizer = tiktoken.encoding_for_model("gpt-4o")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
print("Example text:", text)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("`gpt2` encoded integers:", integers)
strings = tokenizer.decode(integers)
print("`gpt2` decoded strings:", strings)

# Unknown words example.
integers = tokenizer.encode("Akwirw ier")
print("`Akwirw ier` encoded:", integers)
print("Decoded integers:", tokenizer.decode(integers))
