import torch
from model import DecoderOnlyTransformer
from tokenizer import SimpleTokenizer

def load_model(model_path, vocab_size, d_model=256, num_heads=4, n_layers=4):
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        max_len=4096
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def generate_autoregressive(model, input_ids, max_new_tokens=80):
    model.eval()
    device = next(model.parameters()).device

    input_ids = input_ids.to(device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)

        next_token_logits = logits[0, -1]
        next_token = torch.argmax(next_token_logits).item()

        next_token_tensor = torch.tensor([[next_token]], device=device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    return input_ids

def main():
    tokenizer = SimpleTokenizer.from_file("tokenizer.json")
    vocab_size = tokenizer.vocab_size()

    model_path = "model_pretrained.pt"
    model = load_model(model_path, vocab_size)

    with open("input.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    generated_ids = generate_autoregressive(model, input_ids, max_new_tokens=120)

    result = tokenizer.decode(generated_ids[0].tolist())

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print("✔ Modelo executado. Resposta salva em output.txt.")

if __name__ == "__main__":
    main()
