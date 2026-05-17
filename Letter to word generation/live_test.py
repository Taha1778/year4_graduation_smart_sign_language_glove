import torch
import torch.nn as nn
import torch.nn.functional as F
import string

# =========================================================
# CONFIG (must match training)
# =========================================================
MAX_LEN = 8
CONF_THRESHOLD = 0.3   # lower for debugging; raise to 0.7 later

# =========================================================
# LOAD SAVED VOCABULARIES  ✅ MOST IMPORTANT FIX
# =========================================================
try:
    letter_to_idx = torch.load("letter_to_idx.pt")
    word_to_idx = torch.load("word_to_idx.pt")
except FileNotFoundError:
    raise RuntimeError(
        "❌ Missing vocab files.\n"
        "You MUST save letter_to_idx.pt and word_to_idx.pt during training."
    )

idx_to_word = {i: w for w, i in word_to_idx.items()}

# Sanity print (keep for now)
print("✔ Letter vocab size:", len(letter_to_idx))
print("✔ Word vocab size:", len(word_to_idx))

# =========================================================
# MODEL DEFINITION (MUST MATCH TRAINING EXACTLY)
# =========================================================
class LetterToWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_words):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_words)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# =========================================================
# LOAD MODEL WEIGHTS
# =========================================================
model = LetterToWordLSTM(
    vocab_size=len(letter_to_idx),
    embed_dim=32,
    hidden_dim=128,
    num_words=len(word_to_idx),
)

state = torch.load("letter_to_word_model.pt", map_location="cpu")

# Support different save formats
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
elif isinstance(state, dict):
    model.load_state_dict(state)
else:
    model = state  # full model saved

model.eval()

print("✔ Model loaded successfully")

# =========================================================
# ENCODING (IDENTICAL LOGIC TO TRAINING)
# =========================================================
def encode_live_input(word):
    word = word.upper()
    encoded = [letter_to_idx[ch] for ch in word if ch in letter_to_idx]
    encoded = (encoded + [0] * MAX_LEN)[:MAX_LEN]
    return encoded

# Debug helper
print("Encoded HELLO:", encode_live_input("HELLO"))

# =========================================================
# PREDICTION
# =========================================================
def predict_live(word):
    if not word.isalpha():
        return "❌ Invalid input", 0.0

    encoded = encode_live_input(word)
    x = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    conf = conf.item()
    idx = idx.item()

    if conf < CONF_THRESHOLD:
        return "❌ Low confidence – please repeat", conf

    return idx_to_word[idx], conf

# =========================================================
# QUICK SELF-TEST (DO NOT REMOVE)
# =========================================================
print("\n🔎 SELF TEST")
print("HELLO in vocab:", "HELLO" in word_to_idx)
print("Predict HELLO:", predict_live("HELLO"))
print("Predict HELLP:", predict_live("HELLP"))
print("-" * 40)

# =========================================================
# INTERACTIVE LOOP
# =========================================================
def main():
    print("🔤 Letter-to-Word Live Test")
    print("Type a word (or 'exit'):\n")

    while True:
        user_input = input("Input letters: ").strip()
        if user_input.lower() == "exit":
            break

        prediction, confidence = predict_live(user_input)
        print(f"➡️  Prediction: {prediction} (confidence: {confidence:.2f})\n")

if __name__ == "__main__":
    main()