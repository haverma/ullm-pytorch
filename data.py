import os
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import requests
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_FILE = os.path.join(DATA_DIR, "input.txt")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")

def download_and_tokenize():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        print("Downloading Tiny Shakespeare...")
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with open(INPUT_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
    else:
        print("Dataset already downloaded.")

    if not os.path.exists(TRAIN_BIN) or not os.path.exists(VAL_BIN):
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = f.read()
        
        print("Tokenizing using tiktoken (gpt2)...")
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(data, allowed_special={'<|endoftext|>'})
        
        # 90% train, 10% val
        n = len(tokens)
        train_data = tokens[:int(n*0.9)]
        val_data = tokens[int(n*0.9):]
        
        print(f"Train split: {len(train_data)} tokens")
        print(f"Val split: {len(val_data)} tokens")
        
        # Export as numpy array to binary formats. 
        # tiktoken gpt2 vocab is 50257, so uint16 (max 65535) is safe and space-efficient.
        train_data_np = np.array(train_data, dtype=np.uint16)
        val_data_np = np.array(val_data, dtype=np.uint16)
        
        train_data_np.tofile(TRAIN_BIN)
        val_data_np.tofile(VAL_BIN)
        print("Tokenization and caching complete.")
    else:
        print("Dataset already tokenized and cached.")

class LMDataset(Dataset):
    def __init__(self, split, seq_length):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")

        self.seq_length = seq_length
        bin_file = TRAIN_BIN if split == 'train' else VAL_BIN
        if not os.path.exists(bin_file):
            raise FileNotFoundError(
                f"{bin_file} does not exist. Run download_and_tokenize() first."
            )
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        if len(self.data) <= seq_length:
            raise ValueError(
                f"{split} split has {len(self.data)} tokens, which is too short "
                f"for seq_length={seq_length}."
            )

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # We read a block of seq_length + 1 to construct input and target shifts
        chunk = self.data[idx : idx + self.seq_length + 1]
        chunk = chunk.astype(np.int64)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloader(split='train', batch_size=32, seq_length=128, num_workers=0, seed=42):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = LMDataset(split, seq_length)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        generator=g,
        drop_last=True
    )
    return loader

if __name__ == "__main__":
    download_and_tokenize()
    
    # Acceptance check: print one batch's shapes and verify shifted positions
    loader = get_dataloader(split='train', batch_size=2, seq_length=8)
    for x, y in loader:
        print("\n--- Acceptance Check ---")
        print("Input shapes:", x.shape)
        print("Target shapes:", y.shape)
        print("Input tokens (first batch item):", x[0].tolist())
        print("Target tokens (first batch item):", y[0].tolist())
        
        # Determine if target is indeed shifted by one
        is_shifted = (x[0][1:].tolist() == y[0][:-1].tolist())
        print(f"Condition: Target is shifted by exactly one position? {'YES' if is_shifted else 'NO'}")
        break
