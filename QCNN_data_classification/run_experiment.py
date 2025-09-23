#!/usr/bin/env python3
"""
General QCNN experiment runner using the repo's Benchmarking API.
Writes per-ansatz summaries to QCNN/Result/result_ansatz<ansatz>.txt.

Examples (from repo root, next to `QCNN/`):
  python run_experiment.py --dataset mnist --classes 0,1 --ansatz 1 --encoding resize256 --seeds 5
  python run_experiment.py --dataset mnist --classes 0,1 --ansatz 1 --encoding pca8 --seeds 5
"""
import os, sys, argparse, re, statistics, io, datetime

# --- Robust QCNN repo root detection ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_A = os.path.join(SCRIPT_DIR, "QCNN")
CANDIDATE_B = SCRIPT_DIR
def looks_like_qcnn_root(p):
    return os.path.isfile(os.path.join(p, "Benchmarking.py")) and os.path.isdir(os.path.join(p, "Result"))

REPO_QCNN_DIR = CANDIDATE_A if looks_like_qcnn_root(CANDIDATE_A) else (CANDIDATE_B if looks_like_qcnn_root(CANDIDATE_B) else CANDIDATE_A)
if REPO_QCNN_DIR not in sys.path:
    sys.path.insert(0, REPO_QCNN_DIR)

import Benchmarking  # from QCNN/

# Map Table 1 numbering → unitary name and param count
ANSATZ_MAP = {
    "1": ("U_TTN", 2),
    "2": ("U_9", 2),
    "3": ("U_15", 4),
    "4": ("U_13", 6),
    "5": ("U_14", 6),
    "6": ("U_SO4", 6),
    "7": ("U_5", 10),
    "8": ("U_6", 10),
    "9a": ("U_SU4", 15),
    "9b": ("U_SU4_no_pooling", 15),
}
for _k, (_u, _p) in list(ANSATZ_MAP.items()):
    ANSATZ_MAP[_u] = (_u, _p)

# Repo encodings seen in Benchmarking.py
ALLOWED_ENCODINGS = {
    'resize256', 'pca8', 'autoencoder8',
    'pca16', 'autoencoder16', 'pca16-compact',
    'pca32-1','pca32-2','pca32-3','pca32-4',
    'autoencoder32-1','autoencoder32-2','autoencoder32-3','autoencoder32-4',
    'pca30-1','pca30-2','pca30-3','pca30-4',
    'autoencoder30-1','autoencoder30-2','autoencoder30-3','autoencoder30-4',
    'pca16-1','pca16-2','pca16-3','pca16-4',
    'autoencoder16-1','autoencoder16-2','autoencoder16-3','autoencoder16-4',
    'autoencoder12-1','autoencoder12-2','autoencoder12-3','autoencoder12-4',
    'pca12-1','pca12-2','pca12-3','pca12-4',
}

# The repo still appends raw lines to QCNN/Result/result.txt — we parse our own
REPO_RESULTS_FILE = os.path.join(REPO_QCNN_DIR, "Result", "result.txt")

def sanitize_for_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '', str(s))

def main():
    p = argparse.ArgumentParser(description="General QCNN experiment runner (uses QCNN/Benchmarking.py)")
    p.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    p.add_argument("--classes", default="0,1", help="e.g., '0,1', 'even,odd', '<5', '>4'")
    p.add_argument("--ansatz", default="1", help="1..8, 9a, 9b or unitary name (U_TTN, U_SU4, ...)")
    p.add_argument("--encoding", default="resize256", help="Encoding string used by the repo (e.g., resize256, pca8, autoencoder8)")
    p.add_argument("--seeds", type=int, default=5, help="Number of runs")
    p.add_argument("--circuit", choices=["QCNN","Hierarchical"], default="QCNN")
    p.add_argument("--cost_fn", choices=["cross_entropy","mse"], default="cross_entropy",
                   help="For cross_entropy, binary=False; for mse, binary=True.")
    args = p.parse_args()

    # Validate encoding
    if args.encoding not in ALLOWED_ENCODINGS:
        print(f"Warning: encoding '{args.encoding}' not in known list; proceeding anyway. (Repo may still support it.)")

    # Classes parsing
    classes_arg = args.classes.strip()
    if classes_arg in ("even,odd", "odd,even", "even-odd", "odd-even"):
        classes = "odd-even"
    elif classes_arg in ("<5", ">4"):
        classes = classes_arg
    else:
        try:
            a, b = classes_arg.split(",")
            classes = [int(a), int(b)]
        except Exception as exc:
            raise SystemExit(f"Unrecognized --classes: {args.classes}") from exc

    # Map ansatz
    if args.ansatz not in ANSATZ_MAP:
        raise SystemExit(f"Unknown ansatz '{args.ansatz}'. Use 1..8, 9a, 9b or a unitary name present in the repo.")
    unitary_name, u_params = ANSATZ_MAP[args.ansatz]

    encodings = [args.encoding]
    unitaries = [unitary_name]
    u_params_list = [u_params]

    os.makedirs(os.path.join(REPO_QCNN_DIR, "Result"), exist_ok=True)
    # Track growth of the repo's raw results file so we only parse our new lines
    start_size = os.path.getsize(REPO_RESULTS_FILE) if os.path.exists(REPO_RESULTS_FILE) else 0

    binary_flag = (args.cost_fn == "mse")

    # Our per-ansatz summary file
    ansatz_label = sanitize_for_filename(args.ansatz)
    OUT_SUMMARY = os.path.join(REPO_QCNN_DIR, "Result", f"result_ansatz{ansatz_label}.txt")

    print(f"\n=== QCNN experiment ===")
    print(f"Dataset: {args.dataset} | Classes: {args.classes}")
    print(f"Ansatz: {args.ansatz} → {unitary_name} | Encoding: {args.encoding}")
    print(f"Seeds: {args.seeds} | Circuit: {args.circuit} | Cost: {args.cost_fn} | Binary: {binary_flag}")
    print(f"QCNN repo path: {REPO_QCNN_DIR}")
    print(f"Parsing new lines from: {REPO_RESULTS_FILE}")
    print(f"Appending summary to:   {OUT_SUMMARY}\n")

    # Run multiple seeds
    for s in range(args.seeds):
        print(f"[Seed {s+1}/{args.seeds}]")
        Benchmarking.Benchmarking(args.dataset, classes, unitaries, u_params_list, encodings,
                                  circuit=args.circuit, cost_fn=args.cost_fn, binary=binary_flag)

    # Parse the newly appended chunk for our (unitary, encoding)
    new_text = ""
    if os.path.exists(REPO_RESULTS_FILE):
        with open(REPO_RESULTS_FILE, "r", encoding="utf-8") as f:
            f.seek(start_size, io.SEEK_SET)
            new_text = f.read()

    accs = []
    pat = re.compile(rf"Accuracy\s+for\s+{re.escape(unitary_name)}\s+{re.escape(args.encoding)}\s*:(.+)")
    for line in new_text.splitlines():
        m = pat.search(line)
        if m:
            try:
                accs.append(float(m.group(1).strip()))
            except ValueError:
                pass

    if not accs:
        print("\nNo accuracy lines parsed from new results. Check the repo output above.")
        return

    mean_acc = statistics.mean(accs)
    # Use sample std (ddof=1) which is typical for mean±std over seeds
    std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0

    # Print summary to console
    print("\n=== Summary (this invocation) ===")
    print(f"{unitary_name} with {args.encoding} on {args.dataset} {args.classes}:")
    print(f"  Accuracies: {', '.join(f'{a:.4f}' for a in accs)}")
    print(f"  Mean ± Std: {mean_acc:.4f} ± {std_acc:.4f}")

    # Append a clean, single-line record to our per-ansatz file
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"[{ts}] ansatz={args.ansatz} unitary={unitary_name} "
        f"dataset={args.dataset} classes='{args.classes}' encoding={args.encoding} "
        f"seeds={len(accs)} circuit={args.circuit} cost={args.cost_fn} "
        f"accs=[{', '.join(f'{a:.4f}' for a in accs)}] "
        f"mean={mean_acc:.4f} std={std_acc:.4f}\n"
    )
    with open(OUT_SUMMARY, "a", encoding="utf-8") as f:
        f.write(line)

    print(f"\nAppended to {OUT_SUMMARY}")
    print("Done.")

if __name__ == "__main__":
    main()
