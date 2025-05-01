import tiktoken

from classes.dataset import create_dataloader_v1
from data.the_verdict import get_the_verdict

raw_text = get_the_verdict()

tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
print("Encoded text length:", len(enc_text))

# Remove first 50 tokens from the dataset.
enc_sample = enc_text[50:]

# Determines how many tokens are included in the input.
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print(f"x: {x}")
print(f"y:      {y}")

# Depict the `next-word prediction` task.
print("Next-word prediction with token IDs:")
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
print("Next-word prediction with text:")
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# Generate batches with input-target pairs.
dataloader = create_dataloader_v1(
    text=raw_text,
    # batch_size=1,
    batch_size=8,
    max_length=4,
    # max_length=2,
    # max_length=8,
    # stride=1,
    # stride=2,
    # `stride` >= `max_length` -> avoid any overlap between batches -> overlap could lead to increased overfitting.
    stride=4,
    shuffle=False,
)

# Convert to iterator to use the built-in `next()` function.
data_iter = iter(dataloader)

# first_batch = next(data_iter)
# print("1st batch:", first_batch)
# second_batch = next(data_iter)
# print("2nd batch:", second_batch)

inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)
