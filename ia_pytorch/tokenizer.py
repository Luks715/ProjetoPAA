import json
import re
from datasets import load_dataset 

class SimpleTokenizer:

    def __init__(self, token_to_id=None):
        if token_to_id is None:
            self.token_to_id = {}
        else:
            self.token_to_id = token_to_id

        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    # ----------------------------------------
    # 1. Construir vocabulário
    # ----------------------------------------
    def fit(self, text):
        words = re.findall(r"\w+|\S", text.lower())

        self.token_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        idx = len(self.token_to_id)

        for w in words:
            if w not in self.token_to_id:
                self.token_to_id[w] = idx
                idx += 1

        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    # ----------------------------------------
    # 2. Encode
    # ----------------------------------------
    def encode(self, text, add_bos=True, add_eos=True):
        words = re.findall(r"\w+|\S", text.lower())
        ids = []

        if add_bos:
            ids.append(self.token_to_id["<BOS>"])

        for w in words:
            ids.append(self.token_to_id.get(w, self.token_to_id["<UNK>"]))

        if add_eos:
            ids.append(self.token_to_id["<EOS>"])

        return ids

    # ----------------------------------------
    # 3. Decode
    # ----------------------------------------
    def decode(self, ids):
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "<UNK>")
            if tok in self.special_tokens:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    # ----------------------------------------
    # 4. utilitários
    # ----------------------------------------
    def vocab_size(self):
        return len(self.token_to_id)

    def save(self, path):
        with open(path, "w", encoding="utf8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_file(path):
        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        return SimpleTokenizer(token_to_id)

    # ----------------------------------------
    # 5. Builder para usar com o QUATI
    # ----------------------------------------
    @staticmethod
    def build_tokenizer():
        print("Construindo vocabulário...")

        ds = load_dataset("unicamp-dl/quati", split="quati_1M_passages", streaming=True)

        collected = []
        max_chars = 2_000_000   # ~2MB
        size = 0

        for sample in ds:
            text = sample["passage"]
            collected.append(text)
            size += len(text)
            if size >= max_chars:
                break
        
        text = "\n".join(collected)
        
        tokenizer = SimpleTokenizer()
        tokenizer.fit(text)
        tokenizer.save("tokenizer.json")

        print("✔ Vocabulário criado e salvo!")
        return tokenizer
