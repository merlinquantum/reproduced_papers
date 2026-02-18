#!/bin/bash
# Encoding Comparison: Amplitude vs Amplitude+Compression vs Angle
#
# Compares three quantum encoding strategies for adversarial robustness:
# 1. AMPLITUDE (no compression): 256 → 286 Fock states (faithful to paper)
# 2. AMPLITUDE+COMPRESSION: 256 → 64 → 66 Fock states (classical bottleneck)
# 3. ANGLE: 256 → 13 phase shifters (classical encoder required)
#
# This investigates whether classical compression affects adversarial robustness.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PAPER_DIR"

echo "============================================================"
echo "Encoding Comparison: Adversarial Robustness"
echo "============================================================"
echo ""
echo "Comparing three encoding strategies:"
echo "  1. AMPLITUDE:    256 pixels → 286 Fock states (no compression)"
echo "  2. AMP+COMPRESS: 256 pixels → 64 → 66 Fock states"  
echo "  3. ANGLE:        256 pixels → 13 phase shifters"
echo ""

# ============================================================
# 1. TRAINING
# ============================================================
echo "[1/4] Training classifiers..."
echo ""

# Train AMPLITUDE model (no compression)
if [ -f "results/train_quantum/model.pt" ]; then
    echo "--- AMPLITUDE model already exists, skipping training ---"
else
    echo "--- Training AMPLITUDE model (no compression) ---"
    python ../../implementation.py --paper quantum_adversarial_ml \
        --config configs/defaults.json \
        --config configs/train_quantum.json \
        --outdir results/train_quantum
fi

# Train AMPLITUDE+COMPRESSION model
if [ -f "results/train_amplitude_compressed/model.pt" ]; then
    echo "--- AMPLITUDE+COMPRESSION model already exists, skipping training ---"
else
    echo "--- Training AMPLITUDE+COMPRESSION model ---"
    python ../../implementation.py --paper quantum_adversarial_ml \
        --config configs/defaults.json \
        --config configs/train_amplitude_compressed.json \
        --outdir results/train_amplitude_compressed
fi

# Train ANGLE model
if [ -f "results/train_angle/model.pt" ]; then
    echo "--- ANGLE model already exists, skipping training ---"
else
    echo "--- Training ANGLE model ---"
    python ../../implementation.py --paper quantum_adversarial_ml \
        --config configs/defaults.json \
        --config configs/train_angle.json \
        --outdir results/train_angle
fi

echo ""

# ============================================================
# 2. ATTACKS AT EPSILON=0.05
# ============================================================
echo "[2/4] Running BIM attacks at epsilon=0.05..."
echo ""

echo "--- Attacking AMPLITUDE model ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/attack_bim_eps005.json \
    --outdir results/attack_amplitude_eps005

echo "--- Attacking AMPLITUDE+COMPRESSION model ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/attack_ampcomp_eps005.json \
    --outdir results/attack_ampcomp_eps005

echo "--- Attacking ANGLE model ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/attack_angle_eps005.json \
    --outdir results/attack_angle_eps005

echo ""

# ============================================================
# 3. ATTACKS AT EPSILON=0.10
# ============================================================
echo "[3/4] Running BIM attacks at epsilon=0.10..."
echo ""

echo "--- Attacking AMPLITUDE model ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/attack_bim_eps010.json \
    --outdir results/attack_amplitude_eps010

echo "--- Attacking AMPLITUDE+COMPRESSION model ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/attack_ampcomp_eps010.json \
    --outdir results/attack_ampcomp_eps010

echo "--- Attacking ANGLE model ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/attack_angle_eps010.json \
    --outdir results/attack_angle_eps010

echo ""

# ============================================================
# 4. NOISE COMPARISON
# ============================================================
echo "[4/4] Running noise vs adversarial comparison..."
echo ""

echo "--- AMPLITUDE noise comparison ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/noise_comparison.json \
    --outdir results/noise_amplitude

echo "--- AMPLITUDE+COMPRESSION noise comparison ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/noise_comparison_ampcomp.json \
    --outdir results/noise_ampcomp

echo "--- ANGLE noise comparison ---"
python ../../implementation.py --paper quantum_adversarial_ml \
    --config configs/defaults.json \
    --config configs/noise_comparison_angle.json \
    --outdir results/noise_angle

echo ""

# ============================================================
# 5. RESULTS SUMMARY
# ============================================================
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
echo ""

# Extract results using Python for JSON parsing
python3 << 'PYTHON_SCRIPT'
import json
import glob
import os

def find_latest_result(pattern):
    """Find the most recent result file matching pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_json(filepath):
    """Load JSON file safely."""
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath) as f:
                return json.load(f)
        except:
            pass
    return {}

def get_accuracy(data, *keys):
    """Extract accuracy from nested dict."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data

# Collect results for all three encodings
results = {
    'amplitude': {
        'name': 'AMPLITUDE (no compress)',
        'train': find_latest_result('results/train_quantum/*/summary_results.json'),
        'attack_005': find_latest_result('results/attack_amplitude_eps005/*/summary_results.json'),
        'attack_010': find_latest_result('results/attack_amplitude_eps010/*/summary_results.json'),
        'noise': find_latest_result('results/noise_amplitude/*/noise_comparison_results.json'),
    },
    'ampcomp': {
        'name': 'AMPLITUDE+COMPRESS',
        'train': find_latest_result('results/train_amplitude_compressed/*/summary_results.json'),
        'attack_005': find_latest_result('results/attack_ampcomp_eps005/*/summary_results.json'),
        'attack_010': find_latest_result('results/attack_ampcomp_eps010/*/summary_results.json'),
        'noise': find_latest_result('results/noise_ampcomp/*/noise_comparison_results.json'),
    },
    'angle': {
        'name': 'ANGLE',
        'train': find_latest_result('results/train_angle/*/summary_results.json'),
        'attack_005': find_latest_result('results/attack_angle_eps005/*/summary_results.json'),
        'attack_010': find_latest_result('results/attack_angle_eps010/*/summary_results.json'),
        'noise': find_latest_result('results/noise_angle/*/noise_comparison_results.json'),
    }
}

# Load all data
data = {}
for enc_key, enc_info in results.items():
    data[enc_key] = {
        'name': enc_info['name'],
        'train': load_json(enc_info['train']),
        'attack_005': load_json(enc_info['attack_005']),
        'attack_010': load_json(enc_info['attack_010']),
        'noise': load_json(enc_info['noise']),
    }

# Format helper
def fmt(val):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val*100:.1f}%"
    return str(val)

# Extract metrics
metrics = {}
for enc_key in ['amplitude', 'ampcomp', 'angle']:
    d = data[enc_key]
    train_data = d['train']
    atk005_data = d['attack_005']
    atk010_data = d['attack_010']
    
    clean = get_accuracy(train_data, 'best_accuracy')
    adv005 = get_accuracy(atk005_data, 'adversarial_accuracy')
    adv010 = get_accuracy(atk010_data, 'adversarial_accuracy')
    
    metrics[enc_key] = {
        'name': d['name'],
        'clean': clean,
        'adv005': adv005,
        'adv010': adv010,
        'noise': d['noise']
    }

# Print comparison table
print("=" * 85)
print("SIDE-BY-SIDE COMPARISON: THREE ENCODING STRATEGIES")
print("=" * 85)
print()
print(f"{'Metric':<25} {'AMPLITUDE':<20} {'AMP+COMPRESS':<20} {'ANGLE':<20}")
print("-" * 85)

amp = metrics['amplitude']
comp = metrics['ampcomp']
ang = metrics['angle']

print(f"{'Clean Accuracy':<25} {fmt(amp['clean']):<20} {fmt(comp['clean']):<20} {fmt(ang['clean']):<20}")
print(f"{'Adv Acc (ε=0.05)':<25} {fmt(amp['adv005']):<20} {fmt(comp['adv005']):<20} {fmt(ang['adv005']):<20}")
print(f"{'Adv Acc (ε=0.10)':<25} {fmt(amp['adv010']):<20} {fmt(comp['adv010']):<20} {fmt(ang['adv010']):<20}")

# Calculate drops
def calc_drop(clean, adv):
    if clean is not None and adv is not None:
        return f"{(clean - adv) * 100:.1f}%"
    return "N/A"

print(f"{'Accuracy Drop (ε=0.10)':<25} {calc_drop(amp['clean'], amp['adv010']):<20} {calc_drop(comp['clean'], comp['adv010']):<20} {calc_drop(ang['clean'], ang['adv010']):<20}")

# Calculate robustness ratio
def calc_robust(clean, adv):
    if clean is not None and adv is not None and clean > 0:
        return f"{(adv / clean) * 100:.1f}%"
    return "N/A"

print(f"{'Robustness (adv/clean)':<25} {calc_robust(amp['clean'], amp['adv010']):<20} {calc_robust(comp['clean'], comp['adv010']):<20} {calc_robust(ang['clean'], ang['adv010']):<20}")

print()
print("-" * 85)
print("NOISE vs ADVERSARIAL COMPARISON (at ε=0.10)")
print("-" * 85)

for enc_key in ['amplitude', 'ampcomp', 'angle']:
    m = metrics[enc_key]
    noise_data = m['noise']
    if noise_data:
        eps_vals = noise_data.get('epsilon_values', [])
        adv_accs = noise_data.get('adversarial', [])
        uniform_accs = noise_data.get('random_uniform', [])
        
        # Find index for epsilon=0.1
        try:
            idx = eps_vals.index(0.1)
            adv = fmt(adv_accs[idx]) if idx < len(adv_accs) else "N/A"
            uni = fmt(uniform_accs[idx]) if idx < len(uniform_accs) else "N/A"
            print(f"{m['name']:<25} Adversarial: {adv:<12} Random: {uni:<12}")
        except (ValueError, IndexError):
            print(f"{m['name']:<25} (data not available)")
    else:
        print(f"{m['name']:<25} (no noise comparison data)")

print()
print("=" * 85)
print("CONCLUSIONS")
print("=" * 85)
print()

# Check if we have all data
all_have_data = all(
    metrics[k]['clean'] is not None and metrics[k]['adv010'] is not None 
    for k in ['amplitude', 'ampcomp', 'angle']
)

if all_have_data:
    # Find most robust
    robustness = {}
    for k in ['amplitude', 'ampcomp', 'angle']:
        m = metrics[k]
        if m['clean'] and m['clean'] > 0 and m['adv010'] is not None:
            robustness[k] = m['adv010'] / m['clean']
        else:
            robustness[k] = 0
    
    most_robust = max(robustness, key=robustness.get)
    least_robust = min(robustness, key=robustness.get)
    
    print(f"• MOST ROBUST: {metrics[most_robust]['name']} (retains {robustness[most_robust]*100:.1f}% accuracy)")
    print(f"• LEAST ROBUST: {metrics[least_robust]['name']} (retains {robustness[least_robust]*100:.1f}% accuracy)")
    print()
    
    # Check if compression helps
    if robustness['ampcomp'] > robustness['amplitude']:
        diff = (robustness['ampcomp'] - robustness['amplitude']) * 100
        print(f"• COMPRESSION HELPS: Adding classical compression improves robustness by {diff:.1f}%")
    elif robustness['amplitude'] > robustness['ampcomp']:
        diff = (robustness['amplitude'] - robustness['ampcomp']) * 100
        print(f"• COMPRESSION HURTS: Classical compression reduces robustness by {diff:.1f}%")
    else:
        print(f"• COMPRESSION NEUTRAL: No significant difference")
    
    # Compare angle vs amplitude
    if robustness['angle'] > robustness['amplitude']:
        print(f"• ANGLE > AMPLITUDE: Angle encoding is more robust than direct amplitude encoding")
    else:
        print(f"• AMPLITUDE > ANGLE: Direct amplitude encoding is more robust than angle encoding")
else:
    print("(Unable to compute conclusions - some results missing)")

print()

# Save full results to file
summary = {
    'metrics': {k: {
        'name': v['name'],
        'clean_accuracy': v['clean'],
        'adversarial_accuracy_005': v['adv005'],
        'adversarial_accuracy_010': v['adv010'],
    } for k, v in metrics.items()},
}

with open('results/encoding_comparison_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Full results saved to: results/encoding_comparison_summary.json")
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Encoding Comparison Complete!"
echo "============================================================"
