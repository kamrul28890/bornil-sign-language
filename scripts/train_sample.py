"""Train + evaluate the Transformer SLT baseline on the trial sample (WS-C).

End-to-end: load packed sample features + tokenizer, split, train the baseline,
greedy-decode the test split, score with the locked metric protocol, and write a
run folder (config+hash, metrics, predictions) plus an EXPERIMENTS.md row.

    python scripts/train_sample.py --config configs/trial_5pct.yaml
    python scripts/train_sample.py --config configs/trial_5pct.yaml --epochs 3   # smoke
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

from bornil.config import load_config, config_hash
from bornil.benchmark.splits import make_random_split
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.data.transforms import feature_dim
from bornil.models.transformer import SignTransformer
from bornil.train.loop import train_epoch, evaluate


def resolve_device(pref: str) -> str:
    """Resolve 'auto'|'cuda'|'cpu' to a real device, warning on CUDA fallback."""
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but torch.cuda.is_available() is False.\n"
              "         Your torch is likely the CPU build. Install a CUDA build:\n"
              "         pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
              "         Falling back to CPU for now.", flush=True)
        return "cpu"
    return pref


def _subset(manifest, recordings, targets_by_rec):
    rows = manifest[manifest["recording"].isin(set(recordings))]
    pack_idx = rows["pack_idx"].tolist()
    targets = [targets_by_rec[r] for r in rows["recording"]]
    return pack_idx, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trial_5pct.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override for smoke runs")
    ap.add_argument("--out-dir", default=None, help="override trial.out_dir (e.g. smoke)")
    ap.add_argument("--device", default=None, help="auto|cuda|cpu (overrides config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir is not None:
        cfg["trial"]["out_dir"] = args.out_dir
        cfg["trial"]["name"] = os.path.basename(os.path.normpath(args.out_dir))
    if args.device is not None:
        cfg["train"]["device"] = args.device
    out_dir = cfg["trial"]["out_dir"]

    device = resolve_device(cfg["train"]["device"])

    # --- data ---
    man = pd.read_csv(os.path.join(out_dir, "sample_manifest.csv"))
    features = np.load(os.path.join(out_dir, "packed.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(out_dir, "lengths.npy"))
    tok = Tokenizer(cfg["data"]["tokenizer"])
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp = make_random_split(man, seed=cfg["trial"]["seed"], fracs=tuple(cfg["split"]["fracs"]))
    collate = make_collate_fn(tok.pad_id)

    def loader(recs, shuffle):
        pi, tg = _subset(man, recs, targets_by_rec)
        ds = SignDataset(features, lengths, pi, tg,
                         normalize=cfg["data"]["normalize"], upper_body=cfg["data"]["upper_body"])
        # Length-bucketed batches: pad similar-length clips together (less waste).
        bs = LengthBucketBatchSampler(ds.item_lengths(), cfg["train"]["batch_size"],
                                      shuffle=shuffle, seed=cfg["trial"]["seed"])
        return DataLoader(ds, batch_sampler=bs, collate_fn=collate)

    train_loader = loader(sp["train"], True)
    val_loader = loader(sp["val"], False)
    test_loader = loader(sp["test"], False)
    dev_desc = device
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        dev_desc = f"cuda ({p.name}, {p.total_memory/1e9:.1f} GB)"
    print(f"device={dev_desc} | clips train/val/test = "
          f"{len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])} | vocab={tok.vocab_size}")

    # --- model ---
    m = cfg["model"]
    model = SignTransformer(
        input_dim=feature_dim(cfg["data"]["upper_body"]), vocab_size=tok.vocab_size,
        d_model=m["d_model"], nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"], num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"], dropout=m["dropout"], pad_id=tok.pad_id,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    h = config_hash(cfg)
    run_dir = os.path.join(out_dir, "runs", h)
    os.makedirs(run_dir, exist_ok=True)

    # --- train ---
    best_bleu, t0 = -1.0, time.time()
    for ep in range(1, cfg["train"]["epochs"] + 1):
        tr = train_epoch(model, train_loader, opt, device=device, pad_id=tok.pad_id,
                         clip_grad=cfg["train"]["clip_grad"])
        if ep % 5 == 0 or ep == cfg["train"]["epochs"]:
            vm, _, _ = evaluate(model, val_loader, tok, device=device,
                                max_len=cfg["train"]["max_decode_len"])
            print(f"epoch {ep:3d} | train_loss {tr:.4f} | val BLEU-4 {vm['bleu4']:.2f} "
                  f"chrF++ {vm['chrf++']:.2f}", flush=True)
            if vm["bleu4"] > best_bleu:
                best_bleu = vm["bleu4"]
                torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

    # --- test eval ---
    tm, hyps, refs = evaluate(model, test_loader, tok, device=device,
                              max_len=cfg["train"]["max_decode_len"])
    mins = (time.time() - t0) / 60
    print(f"\nTEST | " + " ".join(f"{k}={v:.2f}" for k, v in tm.items()))

    # --- persist run ---
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test": tm, "best_val_bleu4": best_bleu, "train_min": mins}, f, indent=2)
    pd.DataFrame({"reference": refs, "hypothesis": hyps}).to_csv(
        os.path.join(run_dir, "predictions.csv"), index=False, encoding="utf-8")
    _append_experiment_row(cfg, h, tm, device, mins, len(sp["train"]), len(sp["test"]))
    print(f"run saved -> {run_dir}")


def _append_experiment_row(cfg, h, tm, device, mins, n_train, n_test):
    row = (
        f"\n### EXP-{cfg['trial']['name']}-{h}  ({date.today().isoformat()})\n"
        f"- **Model:** SignTransformer d{cfg['model']['d_model']} "
        f"L{cfg['model']['num_encoder_layers']}+{cfg['model']['num_decoder_layers']}   "
        f"**Phase/Task:** {cfg['trial']['name']} (WS-C T0.2)\n"
        f"- **Data/Split:** {cfg['trial']['name']} {cfg['split']['kind']} "
        f"(train {n_train} / test {n_test})\n"
        f"- **Representation:** kp-{feature_dim(cfg['data']['upper_body'])}   "
        f"**Config hash:** {h}   **Seed:** {cfg['trial']['seed']}\n"
        f"- **Compute:** {device}   **Train time:** {mins:.1f} min\n"
        f"- **Metrics (test):** BLEU-4={tm['bleu4']:.2f} chrF={tm['chrf']:.2f} "
        f"chrF++={tm['chrf++']:.2f} ROUGE-L={tm['rougeL']:.3f} "
        f"(BLEU-1/2/3={tm['bleu1']:.1f}/{tm['bleu2']:.1f}/{tm['bleu3']:.1f})\n"
        f"- **Checkpoint:** {cfg['trial']['out_dir']}/runs/{h}/best.pt\n"
    )
    with open(os.path.join("docs", "logs", "EXPERIMENTS.md"), "a", encoding="utf-8") as f:
        f.write(row)


if __name__ == "__main__":
    main()
