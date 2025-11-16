import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from datasets import load_dataset
from model import DecoderOnlyTransformer
from tokenizer import SimpleTokenizer


# -------------------------------------------------------
# IterableDataset para streaming do QUATI
# -------------------------------------------------------

class QuatiStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # streaming=True evita carregar o dataset na RAM
        self.ds = load_dataset("unicamp-dl/quati", split="quati_1M_passages", streaming=True)
    
    def __iter__(self):
        buffer = []

        for sample in self.ds:
            text = sample["passage"]
            tokens = self.tokenizer.encode(text)

            # adicionar tokens ao buffer
            buffer.extend(tokens)

            # enquanto tiver janelas completas, yield
            while len(buffer) >= self.seq_len + 1:
                x = buffer[:self.seq_len]
                y = buffer[1:self.seq_len+1]
                buffer = buffer[self.seq_len:]   # manter fluxo contínuo

                yield (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long)
                )

# -------------------------------------------------------
# Treinamento
# -------------------------------------------------------

def main():
    tokenizer = SimpleTokenizer.from_file("tokenizer.json")
    vocab_size = tokenizer.vocab_size()

    seq_len = 64
    batch_size = 16
    lr = 3e-4
    epochs = 1   # streaming → época "infinita" (cada epoch lê o dataset 1x)

    # Dataset em streaming
    dataset = QuatiStreamingDataset(tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # hiperparâmetros — agora definidos como variáveis
    d_model_value = 256
    n_layers_value = 4
    num_heads_value = 4
    max_len_value = 4096

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model_value,
        n_layers=n_layers_value,
        num_heads=num_heads_value,
        max_len=max_len_value
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Treinando em:", device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------
    # Treino
    # -------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = 0

        # Limita a quantidade de Batches utilizado para treinamento pois o dataset
        # é muito grande e eu não tenho uma GPU, então estou treinando isso em CPU
        max_steps = 20000

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            logits = logits.reshape(-1, vocab_size)
            y = y.reshape(-1)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            # Comente essas duas linhas abaixo para treinar o modelo com TODO o conjunto da Unicamp
            if steps >= max_steps:
                break

        print(f"Loss médio: {total_loss / steps:.4f}")

    # -------------------------------------------------------
    # Salvando o modelo
    # -------------------------------------------------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model_value,
            "n_layers": n_layers_value,
            "num_heads": num_heads_value,
            "max_len": max_len_value
        }
    }, "model_pretrained.pt")
    print("\n✔ Modelo salvo em model_pretrained.pt")
    
    tokenizer.save("tokenizer.json")
    print("✔ Tokenizer salvo em tokenizer.json")


if __name__ == "__main__":
    main()
