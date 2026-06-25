import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

# download COCO images (train2014) if not already present, 12 GB

#!wget http://images.cocodataset.org/zips/train2014.zip -q --show-progress
#!unzip -q train2014.zip

class RefCOCODataset(Dataset):
    def __init__(self, hf_split, image_dir, clip_model="openai/clip-vit-base-patch16"):
        """
        Args:
            hf_split  : one split from load_dataset("jxu124/refcoco")
            image_dir : path to folder containing COCO images
                        e.g. "./train2014"
        """
        self.data      = hf_split
        self.image_dir = image_dir
        self.processor = CLIPProcessor.from_pretrained(clip_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # --- Image ---
        # file_name has a suffix (_16) that doesn't exist on disk, strip it
        raw_info = json.loads(sample["raw_image_info"])
        img_file = raw_info["file_name"]                          # clean filename
        img_path = os.path.join(self.image_dir, img_file)
        image    = Image.open(img_path).convert("RGB")
        W        = raw_info["width"]
        H        = raw_info["height"]

        # --- Text: pick one sentence randomly ---
        sentences = sample["sentences"]                           # list of dicts
        sent      = sentences[np.random.randint(len(sentences))]
        query     = sent["sent"]                                  # clean lowercase text

        # --- BBox: already x1,y1,x2,y2 absolute pixels → normalize ---
        x1, y1, x2, y2 = sample["bbox"]
        bbox = torch.tensor([
            x1 / W,
            y1 / H,
            x2 / W,
            y2 / H,
        ], dtype=torch.float32).clamp(0, 1)

        # --- CLIP preprocessing ---
        encoded = self.processor(
            text=query,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )

        return {
            "pixel_values":   encoded["pixel_values"].squeeze(0),   # (3, 224, 224)
            "input_ids":      encoded["input_ids"].squeeze(0),       # (77,)
            "attention_mask": encoded["attention_mask"].squeeze(0),  # (77,)
            "bbox":           bbox,                                   # (4,) normalized
            "query":          query,                                  # str
        }
