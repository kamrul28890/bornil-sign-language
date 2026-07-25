"""Masked keypoint modeling pretraining (WS-D D1).

Pretrains the encoder half of a SignTransformer (input_proj + TConv frontend
+ Transformer encoder) on unlabeled random-v1 train-split clips, saving an
encoder-only checkpoint scripts/train_sample.py can warm-start from via
--init-encoder.

    python scripts/pretrain_mkm.py --config configs/pretrain_mkm.yaml --device cuda
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bornil.runtime  # noqa: F401  # import torch before pandas (FAIL-002)

import argparse
import json
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bornil.config import config_hash, load_config
from bornil.benchmark.splits import resolve_split
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.models.build import build_model
from bornil.pretrain.checkpoint import save_encoder_checkpoint
from bornil.pretrain.mkm import MaskedSignEncoder
from scripts.train_sample import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_mkm.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override for smoke runs")
    ap.add_argument("--device", default=None, help="auto|cuda|cpu (overrides config)")
    ap.add_argument("--split-root", default="splits")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["pretrain"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["pretrain"]["device"] = args.device
    out_dir = cfg["trial"]["out_dir"]
    device = resolve_device(cfg["pretrain"]["device"])

    man = pd.read_csv(os.path.join(out_dir, "sample_manifest.csv"))
    features = np.load(os.path.join(out_dir, "packed.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(out_dir, "lengths.npy"))
    tok = Tokenizer(cfg["data"]["tokenizer"])
    # Targets are loaded (dataset contract requires them) but never used in
    # the pretraining loss -- no text supervision happens here.
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp, split_source = resolve_split(
        man, kind=cfg["split"]["kind"], seed=cfg["trial"]["seed"],
        fracs=tuple(cfg["split"]["fracs"]), split_root=args.split_root,
    )
    cfg["split"]["source"] = split_source

    rows = man[man["recording"].isin(set(sp["train"]))]
    pack_idx = rows["pack_idx"].tolist()
    targets = [targets_by_rec[r] for r in rows["recording"]]
    ds = SignDataset(features, lengths, pack_idx, targets,
                     normalize=cfg["data"]["normalize"], upper_body=cfg["data"]["upper_body"])
    collate = make_collate_fn(tok.pad_id)
    bs = LengthBucketBatchSampler(ds.item_lengths(), cfg["pretrain"]["batch_size"],
                                  shuffle=True, seed=cfg["trial"]["seed"])
    loader = DataLoader(ds, batch_sampler=bs, collate_fn=collate)

    dev_desc = device
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        dev_desc = f"cuda ({p.name}, {p.total_memory/1e9:.1f} GB)"
    print(f"device={dev_desc} | split={cfg['split']['kind']} train-only ({split_source}) "
          f"| clips={len(sp['train'])}", flush=True)

    backbone = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)
    enc = MaskedSignEncoder(
        backbone, mask_ratio=cfg["pretrain"]["mask_ratio"],
        min_span=cfg["pretrain"]["min_span"], max_span=cfg["pretrain"]["max_span"],
    ).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg["pretrain"]["lr"],
                            weight_decay=cfg["pretrain"]["weight_decay"])

    h = config_hash(cfg)
    run_dir = os.path.join(out_dir, "runs", h)
    os.makedirs(run_dir, exist_ok=True)

    epochs = cfg["pretrain"]["epochs"]
    t0 = time.time()
    losses = []
    for ep in range(1, epochs + 1):
        enc.train()
        total, nb = 0.0, 0
        from tqdm import tqdm
        for x, _y, x_lengths, _y_lengths in tqdm(loader, desc=f"epoch {ep}/{epochs}",
                                                  unit="batch", leave=False, dynamic_ncols=True):
            x = x.to(device)
            x_lengths = x_lengths.to(device)
            opt.zero_grad()
            loss = enc.loss(x, x_lengths)
            loss.backward()
            if cfg["pretrain"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(enc.parameters(), cfg["pretrain"]["clip_grad"])
            opt.step()
            total += float(loss.item())
            nb += 1
        mean_loss = total / max(nb, 1)
        losses.append(mean_loss)
        elapsed = (time.time() - t0) / 60
        print(f"epoch {ep:3d}/{epochs} | recon_loss {mean_loss:.4f} | elapsed {elapsed:.1f}m",
              flush=True)

    save_encoder_checkpoint(os.path.join(run_dir, "encoder_pretrain.pt"), backbone)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "pretrain_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"recon_loss_by_epoch": losses, "train_min": (time.time() - t0) / 60}, f, indent=2)

    _append_pretrain_row(cfg, h, losses, device, (time.time() - t0) / 60, len(sp["train"]))
    print(f"run saved -> {run_dir}")
    print(f"encoder checkpoint -> {os.path.join(run_dir, 'encoder_pretrain.pt')}")


def _append_pretrain_row(cfg, h, losses, device, mins, n_train):
    row = (
        f"\n### EXP-{cfg['trial']['name']}-{h}  ({date.today().isoformat()})\n"
        f"- **Model:** MKM pretrain (encoder-only) d{cfg['model']['d_model']} "
        f"L{cfg['model']['num_encoder_layers']}   **Phase/Task:** {cfg['trial']['name']} (WS-D D1)\n"
        f"- **Data/Split:** {cfg['trial']['name']} {cfg['split']['kind']} train-only "
        f"(train {n_train}); source={cfg['split'].get('source', 'unknown')}\n"
        f"- **Config hash:** {h}   **Seed:** {cfg['trial']['seed']}\n"
        f"- **Compute:** {device}   **Train time:** {mins:.1f} min\n"
        f"- **Pretrain extras:** mask_ratio={cfg['pretrain']['mask_ratio']} "
        f"min_span={cfg['pretrain']['min_span']} max_span={cfg['pretrain']['max_span']}\n"
        f"- **Recon loss:** first={losses[0]:.4f} last={losses[-1]:.4f}\n"
        f"- **Checkpoint:** {cfg['trial']['out_dir']}/runs/{h}/encoder_pretrain.pt\n"
    )
    with open(os.path.join("docs", "logs", "EXPERIMENTS.md"), "a", encoding="utf-8") as f:
        f.write(row)


if __name__ == "__main__":
    main()
