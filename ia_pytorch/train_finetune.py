# Treina apenas com base no PDF com o tema escolhido
# extrair texto do PDF
# tokenizar
# treinar modelo já pré-treinado
# salvar modelo final

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model import DecoderOnlyTransformer
from tokenizer import SimpleTokenizer
from dataset import PDFDataset


def collate_fn(batch):
    input_ids, labels = zip(*batch)
    return torch.stack(input_ids), torch.stack(labels)


def train_finetuning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Treinando em:", device)

    tokenizer = SimpleTokenizer.from_file("tokenizer.json")

    dataset = PDFDataset("manual_para_estudantes_2022.pdf", tokenizer, max_len=256)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = DecoderOnlyTransformer(vocab_size=len(tokenizer.vocab)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"iter {i}, loss {loss.item():.4f}")

    torch.save(model.state_dict(), "finetuned_model.pt")
    print("✔ Fine‑tuning finalizado!")


if __name__ == "__main__":
    train_finetuning()
