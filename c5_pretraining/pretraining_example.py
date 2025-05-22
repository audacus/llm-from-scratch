import tiktoken
import torch

from c4_gpt_model.text_generation import generate_text_simple
from data.config import GPT_CONFIG_124M
from utils.model import GPTModel
from utils.text_tokenizer import text_to_token_ids, token_ids_to_text

# Reduce computational demands by reducing the context length.
config = dict(
    GPT_CONFIG_124M,
    **{
        'context_length': 256,
    },
)

torch.manual_seed(123)
model = GPTModel(config)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Already mapped token IDs.
# ["every effort moves", "I really like"]
inputs = torch.tensor([[16833, 3626, 6100],
                       [40, 1107, 588]])

# The target token IDs we want the model to produce.
# [" effort moves you", " really like chocolate"]
targets = torch.tensor([[3626, 6100, 345],
                        [1107, 588, 11311]])

# 1. Calculate logits.
# Disable gradient tracking, since we are not training.
with torch.no_grad():
    logits = model(inputs)

# 2. Calculate probabilities.
# Probability of each token vocabulary.
probas = torch.softmax(logits, dim=-1)
print("`probas.shape`:", probas.shape)
print("Probabilities:", probas)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

# Print target and effective output to show the difference.
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# 3. Calculate target probabilities.
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# 4. Calculate log probabilities.
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("log probas:", log_probas)

# 5. Calculate average log probability.
avg_log_probas = torch.mean(log_probas)
print("Average log probability:", avg_log_probas)

# 6. Calculate negative average log probability.
neg_avg_log_probas = avg_log_probas * -1
print("Negative average log probability:", neg_avg_log_probas)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

# Same as step 1.-6.
loss = torch.nn.functional.cross_entropy(
    input=logits_flat,
    target=targets_flat,
)
print("Loss:", loss)

perplexity = torch.exp(loss)
print("Perplexity:", perplexity)
