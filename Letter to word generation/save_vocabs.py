import pandas as pd
import string
import torch


# Build letter->idx (PAD=0, A=1..Z=26)
PAD_TOKEN = "<PAD>"
letter_to_idx = {PAD_TOKEN: 0}
for i, l in enumerate(string.ascii_uppercase, start=1):
    letter_to_idx[l] = i

# Read processed dataset and build word->idx from 'target' or 'target_word'
df = pd.read_csv("processed_dataset.csv")
if "target" in df.columns:
    target_col = "target"
elif "target_word" in df.columns:
    target_col = "target_word"
else:
    raise RuntimeError("processed_dataset.csv missing 'target' or 'target_word' column")

unique_words = sorted(df[target_col].astype(str).str.upper().unique())
word_to_idx = {w: i for i, w in enumerate(unique_words)}

# Save to .pt files
torch.save(letter_to_idx, "letter_to_idx.pt")
torch.save(word_to_idx, "word_to_idx.pt")

print(f"Saved letter_to_idx.pt ({len(letter_to_idx)} entries)")
print(f"Saved word_to_idx.pt ({len(word_to_idx)} entries)")
