import re
import torch
from torch.utils.data import Dataset
from pypdf import PdfReader

class PDFDataset(Dataset):
    def __init__(self, pdf_path, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunks = []

        reader = PdfReader(pdf_path)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        full_text = re.sub(r"\s+", " ", full_text)

        tokens = tokenizer.encode(full_text, add_bos=False, add_eos=False)

        for i in range(0, len(tokens) - max_len, max_len):
            x = [tokenizer.token_to_id['<BOS>']] + tokens[i:i+max_len]
            y = tokens[i:i+max_len] + [tokenizer.token_to_id['<EOS>']]
            self.chunks.append((x, y))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        x, y = self.chunks[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# -----------------------------
# Código de execução
# -----------------------------
if __name__ == "__main__":
    from tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer.from_file("tokenizer.json")
    pdf_path = "manual_para_estudantes_2022.pdf"  # Coloque seu PDF aqui
    dataset = PDFDataset(pdf_path, tokenizer, max_len=256)

    # Salva o dataset processado
    torch.save(dataset, "dataset_finetune.pt")
    print(f"✔ Dataset salvo em dataset_finetune.pt com {len(dataset)} amostras")
