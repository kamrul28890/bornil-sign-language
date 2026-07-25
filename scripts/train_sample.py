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
from bornil.benchmark.splits import (
    filter_split_by_fraction,
    filter_split_by_recordings,
    resolve_split,
)
from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.data.sampler import LengthBucketBatchSampler
from bornil.data.tokenizer import Tokenizer
from bornil.data.transforms import feature_dim
from bornil.models.build import build_model
from bornil.pretrain.checkpoint import load_encoder_checkpoint
from bornil.train.checkpoint import load_training_state, save_training_state
from bornil.train.loop import train_epoch, evaluate
from bornil.train.schedule import noam_lambda


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


def _apply_target_length_filter(cfg, manifest, split, split_source):
    max_tokens = cfg.get("data", {}).get("max_target_tokens")
    if max_tokens is None:
        return split, split_source
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise SystemExit("data.max_target_tokens must be a positive integer")

    lengths = manifest["clean_text"].astype(str).str.split().map(len)
    allowed = set(manifest.loc[lengths <= max_tokens, "recording"])
    filtered = filter_split_by_recordings(split, allowed)
    before = {name: len(split.get(name, [])) for name in ("train", "val", "test")}
    after = {name: len(filtered.get(name, [])) for name in ("train", "val", "test")}
    if any(after[name] == 0 for name in ("train", "val", "test")):
        raise SystemExit(
            "target-length filter removed an entire split partition: "
            f"before={before} after={after}"
        )

    cfg.setdefault("split", {})["filter"] = {
        "max_target_tokens": max_tokens,
        "before": before,
        "after": after,
    }
    split_source = f"{split_source}|max_target_tokens<={max_tokens}"
    print(f"target_length_filter<={max_tokens} tokens | before={before} | after={after}")
    return filtered, split_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/trial_5pct.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override for smoke runs")
    ap.add_argument("--out-dir", default=None, help="override trial.out_dir (e.g. smoke)")
    ap.add_argument("--device", default=None, help="auto|cuda|cpu (overrides config)")
    ap.add_argument("--beam-size", type=int, default=None,
                    help="override train.beam_size for final test decode")
    ap.add_argument("--split", default=None,
                    help="override split.kind: random-v1|signerindep-v1|unseen-sentence-v1")
    ap.add_argument("--split-root", default="splits",
                    help="directory containing frozen split files")
    ap.add_argument("--no-frozen-splits", action="store_true",
                    help="generate the split from the current manifest instead of loading splits/<kind>")
    ap.add_argument("--resume", action="store_true",
                    help=("resume from <run_dir>/last.pt when present. Execution-only; "
                          "does not enter the config hash. Batch-sampler RNG state is "
                          "not restored, so resumed epochs may use a different batch order."))
    ap.add_argument("--init-encoder", default=None,
                    help="path to an encoder_pretrain.pt from pretrain_mkm.py; "
                         "warm-starts input_proj/temporal_frontend/transformer.encoder")
    ap.add_argument("--label-fraction", type=float, default=None,
                    help="subsample the train split to this fraction of clips "
                         "(data-efficiency curve; val/test unaffected)")
    ap.add_argument("--label-fraction-seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # --resume is intentionally argparse-only: it is an execution mode, not a
    # hyperparameter. If it entered cfg, config_hash(cfg) would change and the
    # run would look in a different run_dir for its own checkpoint.
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir is not None:
        cfg["trial"]["out_dir"] = args.out_dir
        cfg["trial"]["name"] = os.path.basename(os.path.normpath(args.out_dir))
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.beam_size is not None:
        cfg["train"]["beam_size"] = args.beam_size
    if args.split is not None:
        cfg.setdefault("split", {})["kind"] = args.split
    out_dir = cfg["trial"]["out_dir"]

    device = resolve_device(cfg["train"]["device"])

    # --- data ---
    man = pd.read_csv(os.path.join(out_dir, "sample_manifest.csv"))
    features = np.load(os.path.join(out_dir, "packed.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(out_dir, "lengths.npy"))
    tok = Tokenizer(cfg["data"]["tokenizer"])
    targets_by_rec = {r: tok.encode(t) for r, t in zip(man["recording"], man["clean_text"])}

    sp, split_source = resolve_split(
        man,
        kind=cfg["split"]["kind"],
        seed=cfg["trial"]["seed"],
        fracs=tuple(cfg["split"]["fracs"]),
        split_root=args.split_root,
        prefer_frozen=not args.no_frozen_splits,
    )
    sp, split_source = _apply_target_length_filter(cfg, man, sp, split_source)
    if args.label_fraction is not None:
        sp = filter_split_by_fraction(sp, args.label_fraction, seed=args.label_fraction_seed)
        split_source = f"{split_source}|label_fraction={args.label_fraction}"
        cfg.setdefault("split", {})["label_fraction"] = args.label_fraction
        print(f"label_fraction={args.label_fraction} -> train clips={len(sp['train'])}", flush=True)
    cfg["split"]["source"] = split_source
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
    print(f"device={dev_desc} | split={cfg['split']['kind']} ({split_source}) | "
          f"clips train/val/test = {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])} "
          f"| vocab={tok.vocab_size}")

    # --- model ---
    # Built through the shared builder so offline scripts rebuild this exact
    # model from config.json (see bornil/models/build.py).
    m = cfg["model"]
    arch = m.get("arch", "transformer")
    model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)
    if args.init_encoder:
        loaded = load_encoder_checkpoint(model, args.init_encoder, device)
        print(f"init_encoder: loaded {len(loaded)} tensors from {args.init_encoder}", flush=True)
        cfg.setdefault("train", {})["init_encoder"] = args.init_encoder
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    # Warmup+decay LR schedule (stepped per optimizer step). Default 6% of total
    # steps as warmup, capped at half so short trials still decay.
    epochs = cfg["train"]["epochs"]
    total_steps = max(1, len(train_loader) * epochs)
    warmup = int(cfg["train"].get("warmup_steps", 0)) or max(1, int(0.06 * total_steps))
    warmup = min(warmup, max(1, total_steps // 2))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam_lambda(warmup))
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    early_eos_penalty = float(cfg["train"].get("early_eos_penalty", 0.0))
    repetition_stop = int(cfg["train"].get("decode_repetition_stop", 0))
    ctc_weight = float(cfg["train"].get("ctc_weight", 0.0))
    beam_size = int(cfg["train"].get("beam_size", 1))
    decode_mode = "greedy" if beam_size <= 1 else f"beam{beam_size}"
    cfg["train"]["decode_mode"] = decode_mode
    print(f"warmup_steps={warmup}/{total_steps} | label_smoothing={label_smoothing} "
          f"| early_eos_penalty={early_eos_penalty} | decode={decode_mode} "
          f"| repetition_stop={repetition_stop} | ctc_weight={ctc_weight}", flush=True)

    h = config_hash(cfg)
    run_dir = os.path.join(out_dir, "runs", h)
    os.makedirs(run_dir, exist_ok=True)
    last_ckpt = os.path.join(run_dir, "last.pt")

    best_bleu = -1.0
    start_epoch = 1
    elapsed_prior_min = 0.0
    if args.resume:
        if os.path.exists(last_ckpt):
            state = load_training_state(last_ckpt, model, opt, sched, device)
            best_bleu = state["best_bleu"]
            elapsed_prior_min = state["elapsed_min"]
            start_epoch = state["epoch"] + 1
            print(
                f"resumed from {last_ckpt} at epoch {state['epoch']}/{epochs}; "
                f"next epoch {start_epoch}/{epochs}; prior elapsed {elapsed_prior_min:.1f}m",
                flush=True,
            )
        else:
            print(f"--resume requested but no checkpoint found at {last_ckpt}; starting fresh",
                  flush=True)

    # --- train ---
    t0 = time.time()
    if start_epoch > epochs:
        print(f"training already complete at epoch {start_epoch - 1}/{epochs}; "
              "skipping to test eval", flush=True)
    for ep in range(start_epoch, epochs + 1):
        tr = train_epoch(model, train_loader, opt, device=device, pad_id=tok.pad_id,
                         clip_grad=cfg["train"]["clip_grad"],
                         label_smoothing=label_smoothing, scheduler=sched,
                         progress=True, desc=f"epoch {ep}/{epochs}",
                         eos_id=tok.eos_id, early_eos_penalty=early_eos_penalty,
                         ctc_weight=ctc_weight, bos_id=tok.bos_id)
        session_elapsed = time.time() - t0
        elapsed_min = elapsed_prior_min + session_elapsed / 60
        completed_this_session = ep - start_epoch + 1
        eta = session_elapsed / max(completed_this_session, 1) * (epochs - ep)
        line = (f"epoch {ep:3d}/{epochs} | loss {tr:.4f} | lr {opt.param_groups[0]['lr']:.2e} "
                f"| elapsed {elapsed_min:.1f}m | ETA {eta/60:.1f}m")
        if ep % 5 == 0 or ep == epochs:
            vm, _, _ = evaluate(model, val_loader, tok, device=device,
                                max_len=cfg["train"]["max_decode_len"],
                                repetition_stop=repetition_stop)  # greedy for speed
            line += f" | val BLEU-4 {vm['bleu4']:.2f} chrF++ {vm['chrf++']:.2f}"
            if vm["bleu4"] > best_bleu:
                best_bleu = vm["bleu4"]
                torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        save_training_state(last_ckpt, ep, model, opt, sched, best_bleu, elapsed_min)
        print(line, flush=True)

    # --- test eval (best checkpoint, beam search) ---
    best_ckpt = os.path.join(run_dir, "best.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    tm, hyps, refs = evaluate(model, test_loader, tok, device=device,
                              max_len=cfg["train"]["max_decode_len"], beam_size=beam_size,
                              repetition_stop=repetition_stop,
                              progress=True, desc=f"test decode ({decode_mode})")
    mins = elapsed_prior_min + (time.time() - t0) / 60
    print(f"\nTEST ({decode_mode}) | " + " ".join(f"{k}={v:.2f}" for k, v in tm.items()))

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
    front = ""
    if int(cfg["model"].get("temporal_conv_layers", 0)):
        front = (f" + TConv{cfg['model']['temporal_conv_layers']}"
                 f"k{cfg['model'].get('temporal_conv_kernel', 5)}")
    arch = cfg["model"].get("arch", "transformer")
    if arch == "bilstm":
        model_name = "BiLSTM"
        enc_layers = cfg["model"].get("enc_layers", "?")
        phase_task = "WS-C T0.1"
    elif arch == "stgcn":
        model_name = "ST-GCN"
        enc_layers = cfg["model"].get("stgcn_layers", "?")
        phase_task = "WS-C T1.1"
    else:
        model_name = "SignTransformer"
        enc_layers = cfg["model"].get("num_encoder_layers", "?")
        phase_task = "WS-C T0.2"
    row = (
        f"\n### EXP-{cfg['trial']['name']}-{h}  ({date.today().isoformat()})\n"
        f"- **Model:** {model_name}{front} d{cfg['model']['d_model']} "
        f"L{enc_layers}+{cfg['model']['num_decoder_layers']}   "
        f"**Phase/Task:** {cfg['trial']['name']} ({phase_task})\n"
        f"- **Data/Split:** {cfg['trial']['name']} {cfg['split']['kind']} "
        f"(train {n_train} / test {n_test}); source={cfg['split'].get('source', 'unknown')}\n"
        f"- **Representation:** kp-{feature_dim(cfg['data']['upper_body'])}   "
        f"**Config hash:** {h}   **Seed:** {cfg['trial']['seed']}\n"
        f"- **Compute:** {device}   **Train time:** {mins:.1f} min\n"
        f"- **Decode:** {cfg['train'].get('decode_mode', 'unknown')} "
        f"(beam_size={cfg['train'].get('beam_size', 1)})\n"
        f"- **Training extras:** early_eos_penalty={cfg['train'].get('early_eos_penalty', 0.0)} "
        f"ctc_weight={cfg['train'].get('ctc_weight', 0.0)} "
        f"init_encoder={cfg['train'].get('init_encoder', 'none')} "
        f"label_fraction={cfg.get('split', {}).get('label_fraction', 1.0)}\n"
        f"- **Decode extras:** decode_repetition_stop={cfg['train'].get('decode_repetition_stop', 0)}\n"
        f"- **Metrics (test):** BLEU-4={tm['bleu4']:.2f} chrF={tm['chrf']:.2f} "
        f"chrF++={tm['chrf++']:.2f} ROUGE-L={tm['rougeL']:.3f} "
        f"(BLEU-1/2/3={tm['bleu1']:.1f}/{tm['bleu2']:.1f}/{tm['bleu3']:.1f})\n"
        f"- **Checkpoint:** {cfg['trial']['out_dir']}/runs/{h}/best.pt\n"
    )
    with open(os.path.join("docs", "logs", "EXPERIMENTS.md"), "a", encoding="utf-8") as f:
        f.write(row)


if __name__ == "__main__":
    main()
