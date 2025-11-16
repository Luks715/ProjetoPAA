import torch
from pathlib import Path
from modelo_pytorch.model import DecoderOnlyTransformer
from modelo_pytorch.tokenizer import SimpleTokenizer

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "modelo_pytorch" / "model_pretrained.pt"
TOKENIZER_PATH = BASE_DIR / "modelo_pytorch" / "tokenizer.json"

# Seleção automática de dispositivo para rodar a IA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"[IA] Carregando modelo em: {device}")

# a função torch.cuda.is_avaliable() verifica 3 coisas: 
# 1: se o cuda está instalado; 
# 2: se existem drivers da nvidia no dispositivo; 
# 3: se existe uma gpu nvidia física detectada

# Carrega tokenizer
tokenizer = SimpleTokenizer.from_file(str(TOKENIZER_PATH))

# Carrega modelo
checkpoint = torch.load(MODEL_PATH, map_location= device)
config = checkpoint["config"]

model = DecoderOnlyTransformer(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_layers=config["n_layers"],
    num_heads=config["num_heads"],
    max_len=config["max_len"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Função de geração
def generate_response(prompt, max_new_tokens=80):
    # Gera texto usando o modelo carregado

    # prepara entrada para o device
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], 
        dtype=torch.long
    ).to(device)

    model.eval()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1]).item()

        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    return tokenizer.decode(input_ids[0].tolist())