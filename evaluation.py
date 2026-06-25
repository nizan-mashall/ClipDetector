from model import DetectorHead
import torch
import clip
from dataloader import RefCOCODataset
from torch.utils.data import DataLoader
import json
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tqdm
from datasets import load_dataset


dataset = load_dataset("jxu124/refcoco")

train_ds = RefCOCODataset(dataset["train"],      image_dir="./train2014")
val_ds   = RefCOCODataset(dataset["validation"], image_dir="./train2014")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

def predict(image, query, clip_model, head, preprocess, device):
    """
    Args:
        image  : PIL Image
        query  : str — natural language description
    Returns:
        bbox   : (4,) numpy array — normalized (x1,y1,x2,y2)
    """
    head.eval()

    # encode
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([query]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        text_features  = clip_model.encode_text(text_tokens).float()

    # normalize + fuse
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
    fused          = torch.cat([image_features, text_features], dim=-1)

    # predict
    with torch.no_grad():
        bbox = head(fused).squeeze(0).cpu().numpy()  # (4,)

    return bbox


def show_prediction(image, pred_bbox, query, gt_bbox=None):
    """
    Draw predicted box (red) and optionally ground truth box (green).
    pred_bbox / gt_bbox: normalized (x1,y1,x2,y2)
    """
    W, H = image.size
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)

    # predicted box (red dashed)
    x1, y1, x2, y2 = pred_bbox * np.array([W, H, W, H])
    ax.add_patch(patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=2, edgecolor="red",
        facecolor="none", linestyle="--", label="Predicted"
    ))

    # ground truth box (green) if provided
    if gt_bbox is not None:
        x1, y1, x2, y2 = gt_bbox * np.array([W, H, W, H])
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="limegreen",
            facecolor="none", label="Ground truth"
        ))

    ax.set_title(f'Query: "{query}"', fontsize=11)
    ax.legend(loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def compute_iou(pred, gt):
    """Compute IoU between two normalized (x1,y1,x2,y2) boxes."""
    ix1 = max(pred[0], gt[0])
    iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2])
    iy2 = min(pred[3], gt[3])

    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    pred_area = (pred[2]-pred[0]) * (pred[3]-pred[1])
    gt_area   = (gt[2]-gt[0])   * (gt[3]-gt[1])
    union     = pred_area + gt_area - inter

    return inter / union if union > 0 else 0.0


# ── Test on a single val sample ───────────────────────────────────────────────
for idx in range(len(dataset["validation"])):
  sample   = dataset["validation"][idx]
  raw_info = json.loads(sample["raw_image_info"])
  W, H     = raw_info["width"], raw_info["height"]

  image    = Image.open(f"./train2014/{raw_info['file_name']}").convert("RGB")
  query    = sample["sentences"][0]["sent"]
  x1,y1,x2,y2 = sample["bbox"]
  gt_bbox  = np.array([x1/W, y1/H, x2/W, y2/H])

  pred_bbox = predict(image, query, clip_model, head, preprocess, DEVICE)
  iou       = compute_iou(pred_bbox, gt_bbox)
  
  if iou > 0.7:
    print(f"Query:     '{query}'")
    print(f"Predicted: {pred_bbox.round(3)}")
    print(f"GT:        {gt_bbox.round(3)}")
    print(f"IoU:       {iou:.4f}")

    show_prediction(image, pred_bbox, query, gt_bbox)
    break


# ── Evaluate on N val samples ─────────────────────────────────────────────────
def evaluate(dataset_split, n=200):
    ious = []
    for i in tqdm(range(n), desc="Evaluating"):
        sample   = dataset_split[i]
        raw_info = json.loads(sample["raw_image_info"])
        W, H     = raw_info["width"], raw_info["height"]

        try:
            image = Image.open(f"./train2014/{raw_info['file_name']}").convert("RGB")
        except FileNotFoundError:
            continue

        query   = sample["sentences"][0]["sent"]
        x1,y1,x2,y2 = sample["bbox"]
        gt_bbox = np.array([x1/W, y1/H, x2/W, y2/H])

        pred_bbox = predict(image, query, clip_model, head, preprocess, DEVICE)
        ious.append(compute_iou(pred_bbox, gt_bbox))

    mean_iou  = np.mean(ious)
    acc_50    = np.mean([iou >= 0.5 for iou in ious])  # standard grounding metric
    acc_25    = np.mean([iou >= 0.25 for iou in ious])

    print(f"\nResults over {len(ious)} samples:")
    print(f"  Mean IoU : {mean_iou:.4f}")
    print(f"  Acc@0.50 : {acc_50:.4f}  (pred overlaps GT by >50%)")
    print(f"  Acc@0.25 : {acc_25:.4f}  (pred overlaps GT by >25%)")
    return mean_iou, acc_50

if __name__ == "__main__":
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    head = DetectorHead(input_dim=1024, hidden_dim=512).to(DEVICE)

    ckpt = torch.load("./checkpoints/detector_head.pt", map_location=DEVICE)
    head.load_state_dict(ckpt["model_state"])
    head.eval()

    clip_model, preprocess = clip.load("ViT-B/16", device=DEVICE)
    clip_model.eval()

    evaluate(dataset["validation"], n=200)