import tiktoken
import torch

from data.config import GPT_CONFIG_124M
from utils.model import GPTModel, generate_text_simple

start_context = "Hello, I am"

tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
print("Encoded:", encoded)
# Add batch dimension.
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("`encoded_tensor.shape`:", encoded_tensor.shape)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
# Disable dropout since we are not training the model.
model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)