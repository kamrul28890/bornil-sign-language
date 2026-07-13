"""One-command trial: sample -> extract -> pack -> train+eval, GPU if available.

Enter a percentage of the dataset and it runs the whole vertical slice, streaming
each step's progress. Uses the GPU automatically (device: auto); pass --device to
force it. Resumable: re-running skips already-extracted clips.

    python scripts/run_trial.py --percent 2                  # 2% end to end
    python scripts/run_trial.py --percent 5 --device cuda
    python scripts/run_trial.py --percent 1 --epochs 20      # fewer epochs
    python scripts/run_trial.py --percent 2 --skip-extract   # reuse features, retrain
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bornil.trial import trial_dirname

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def run(step: str, script: str, argv):
    print(f"\n{'=' * 64}\n### {step}\n{'=' * 64}", flush=True)
    subprocess.run([sys.executable, os.path.join(SCRIPTS, script)] + argv, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percent", type=float, default=2.0,
                    help="percentage of the dataset for the trial (e.g. 2)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel CPU workers for extraction (default cpu_count-2)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--config", default="configs/trial_base.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sample", action="store_true")
    ap.add_argument("--skip-extract", action="store_true",
                    help="reuse already-extracted features (retrain only)")
    args = ap.parse_args()

    out = trial_dirname(args.percent)
    print(f"TRIAL {args.percent:g}%  ->  {out}  (device={args.device})", flush=True)

    if not args.skip_sample:
        run("1/4 Sample clips", "make_sample.py",
            ["--percent", str(args.percent), "--out-dir", out, "--seed", str(args.seed)])
    if not args.skip_extract:
        extract_argv = ["--out-dir", out]
        if args.workers is not None:
            extract_argv += ["--workers", str(args.workers)]
        run("2/4 Extract MediaPipe keypoints (CPU-parallel)", "extract_sample.py",
            extract_argv)
    run("3/4 Pack features", "pack_sample.py", ["--out-dir", out])

    train_argv = ["--config", args.config, "--out-dir", out, "--device", args.device]
    if args.epochs is not None:
        train_argv += ["--epochs", str(args.epochs)]
    run("4/4 Train + evaluate (GPU if available)", "train_sample.py", train_argv)

    print(f"\n{'=' * 64}\nTRIAL COMPLETE -> {out}/  (see docs/logs/EXPERIMENTS.md)\n{'=' * 64}")


if __name__ == "__main__":
    main()
