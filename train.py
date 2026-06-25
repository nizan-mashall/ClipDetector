import torch
import clip
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import DetectorHead
from dataloader import RefCOCODataset

from datasets import load_dataset

dataset = load_dataset("jxu124/refcoco")

train_ds = RefCOCODataset(dataset["train"],      image_dir="./train2014")
val_ds   = RefCOCODataset(dataset["validation"], image_dir="./train2014")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)


# ── Setup (outside loops) ────────────────────────────────────────────────────
EPOCHS   = 10
LR       = 3e-4
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/16", device=DEVICE)
clip_model.eval()

head      = DetectorHead(input_dim=1024, hidden_dim=512).to(DEVICE)
optimizer = Adam(head.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()   # better than MSE for bbox

# ── Helpers ──────────────────────────────────────────────────────────────────
def encode_batch(pixel_values, input_ids, attention_mask):
    """Extract and fuse CLIP features for a whole batch."""
    with torch.no_grad():
        image_features = clip_model.encode_image(pixel_values).float()
        text_features  = clip_model.encode_text(input_ids).float()

    # normalize before concat
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

    return torch.cat([image_features, text_features], dim=-1)  # (B, 1024)


# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    head.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        pixel_values   = batch["pixel_values"].to(DEVICE)
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        targets        = batch["bbox"].to(DEVICE)           # (B, 4)

        fused       = encode_batch(pixel_values, input_ids, attention_mask)
        predictions = head(fused)                           # (B, 4)

        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Average loss: {avg_loss}")



    # ── Validation ───────────────────────────────────────────────────────────
    head.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values   = batch["pixel_values"].to(DEVICE)
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets        = batch["bbox"].to(DEVICE)

            fused       = encode_batch(pixel_values, input_ids, attention_mask)
            predictions = head(fused)
            val_loss   += criterion(predictions, targets).item()

    avg_val = val_loss / len(val_loader)
    print(f"Epoch {epoch+1:02d} | train_loss={avg_loss:.4f} | val_loss={avg_val:.4f}")


# ── Save ─────────────────────────────────────────────────────────────────────

os.makedirs("./checkpoints", exist_ok=True)

torch.save({
    "epoch":       epoch,
    "model_state": head.state_dict(),
    "optimizer":   optimizer.state_dict(),
    "val_loss":    avg_val,
}, "./checkpoints/detector_head.pt")

print("Model saved to ./checkpoints/detector_head.pt")