"""
Pointing Gesture Target Selection Experiment Analysis

This script analyzes pointing gesture experiments with three target anchors (5C19, DC0F, 96BB).
The study evaluates different methods for target selection across various experimental conditions:

EXPERIMENTAL CONDITIONS:
1. Baseline: User positioned in center of anchor triangle
2. Proximity: User is positioned ~75cm away from DC0F anchor
3. Obstruction: DC0F anchor occluded by another person
4. Anchor Spacing: Anchors 5C19 & 96BB repositioned at 2m, 1m, 0.5m intervals

METHODS TESTED:
- IMU_Bearing: IMU-based bearing/pointing direction calculation alone
- UWB_Distance: UWB distance change detection alone during gesture
- Sensor_Fusion: Final selection using margin score to combine IMU + UWB methods

RESEARCH FOCUS AREAS:
1. Benchmark performance evaluation
2. Target proximity and spacing variation effects
3. Sensor fusion comparison across conditions
"""

import os, re, math, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Style for papers =====
sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

FOLDER = "../results"        # <- set to your folder (e.g., "../results")
SEQ_LEN = 20

GROUND_TRUTH = [
    "5C19","DC0F","96BB","5C19","DC0F",
    "96BB","DC0F","5C19","DC0F","5C19",
    "96BB","5C19","5C19","DC0F","96BB",
    "96BB","DC0F","96BB","DC0F","5C19"
]

# Method mapping: maps readable names to CSV column names
METHOD_COLUMNS = {
    "IMU Bearing":   "bearing",           # IMU-based bearing calculation (pointing direction)
    "UWB Distance":  "distance_change",   # UWB distance change method
    "Sensor Fusion": "selection",  # Sensor fusion score-based selection
}
METHODS = ["IMU Bearing", "UWB Distance", "Sensor Fusion"]


EXPERIMENTS = {
    1: {"name": "Baseline", "category": "benchmark", "description": "User in center of anchor triangle"},
    2: {"name": "User Position", "category": "user_position", "description": "User moved ~75cm toward DC0F anchor"},
    3: {"name": "Obstruction", "category": "environmental", "description": "DC0F anchor occluded by person"},
    4: {"name": "Anchor Spacing", "category": "anchor_spacing", "description": "Anchors 5C19 & 96BB repositioned closer"},
}
# Exp-4 mapping via filename IDs:
EXP4_ID_TO_SPACING = {41: "2 m", 42: "1 m", 43: "0.5 m"}  # labels for plots

LABELS = ["5C19","DC0F","96BB"]
OCCLUDED_ANCHOR_EXP3 = "DC0F"

def _find_col(df, column_name):
    """Check if a column exists in the dataframe"""
    if column_name in df.columns:
        return column_name
    return None

def _parse_from_filename(fname):
    """
    '<expId>_<run>_selections.csv'
    expId in {1,2,3} or {41,42,43} where 41/42/43 map to Exp-4 spacings.
    """
    m = re.match(r"^(\d+)_", fname)
    if not m: return None, None, None
    raw = int(m.group(1))
    if raw in (1,2,3):
        return raw, None, None
    if raw in EXP4_ID_TO_SPACING:
        return 4, raw, EXP4_ID_TO_SPACING[raw]
    return None, None, None


# =======================
# Load & reshape data for analysis
# =======================
gesture_selection_data = []  # Individual pointing gesture results across all experiments
experiment_run_accuracy = []  # Accuracy metrics per experimental run

files = [f for f in os.listdir(FOLDER) if f.endswith(".csv")]
if not files:
    raise SystemExit(f"No CSVs in {FOLDER}")

for fname in sorted(files):
    exp, exp4_id, spacing = _parse_from_filename(fname)
    if exp is None:
        continue
    df = pd.read_csv(os.path.join(FOLDER, fname))
    if len(df) % SEQ_LEN != 0:
        warnings.warn(f"{fname}: {len(df)} rows not multiple of {SEQ_LEN}")

    # map method columns
    colmap = {method: _find_col(df, col) for method, col in METHOD_COLUMNS.items()}
    # keep only available methods
    present_methods = [method for method in METHODS if colmap[method] is not None]

    # break file into 20-selection runs
    num_runs = len(df) // SEQ_LEN
    for run_idx, chunk in enumerate(np.array_split(df, num_runs)):
        gt = np.array(GROUND_TRUTH)
        for method in present_methods:
            preds = chunk[colmap[method]].astype(str).values
            correct = (preds == gt).astype(int)
            acc = correct.mean()
            experiment_run_accuracy.append({
                "file": fname, "experiment": exp,
                "experiment_name": EXPERIMENTS[exp]["name"],
                "spacing": spacing if exp==4 else None,
                "method": method, "run": run_idx, "accuracy": acc
            })
            # individual gesture results per selection (1..20)
            for i in range(SEQ_LEN):
                gesture_selection_data.append({
                    "file": fname, "experiment": exp,
                    "experiment_name": EXPERIMENTS[exp]["name"],
                    "exp4_spacing": spacing if exp==4 else None,
                    "method": method,
                    "selection_index": i+1,          # 1..20 for plots
                    "pred": preds[i],
                    "true": gt[i],
                    "correct": int(preds[i] == gt[i]),
                    "run": run_idx
                })

accuracy_metrics = pd.DataFrame(experiment_run_accuracy)
selection_results = pd.DataFrame(gesture_selection_data)
if accuracy_metrics.empty or selection_results.empty:
    raise SystemExit("No usable rows; check filenames/columns.")

# =======================
# Utility: confusion & pairwise confusion metric
# =======================
def confusion_matrix_from_df(df_sub, labels=LABELS):
    t = df_sub["true"].values
    p = df_sub["pred"].values
    idx = {lab:i for i,lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a,b in zip(t,p):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1
    return cm

def pair_confusion_rate(df_sub, a="5C19", b="96BB"):
    """Symmetric error rate between a and b: (a→b + b→a) / (all a or b true)"""
    mask_ab = df_sub["true"].isin([a,b])
    if mask_ab.sum() == 0: return np.nan
    mis_ab = ((df_sub["true"]==a) & (df_sub["pred"]==b)).sum() + \
             ((df_sub["true"]==b) & (df_sub["pred"]==a)).sum()
    return mis_ab / mask_ab.sum()

# =======================
# FIGURE 1 — Accuracy over selections (1–20) 
# =======================
def fig1_accuracy_over_selections(out_png="../plots/linecharts/fig1_selections.png", out_svg="../plots/linecharts/fig1_selections.svg"):
    df = (selection_results.groupby(["method","selection_index","file","run"], as_index=False)
               .agg(correct=("correct","mean")))
    # Convert to percentage
    df["correct"] = df["correct"] * 100
    
    # Plot means only, no confidence intervals, with specified order
    g = sns.lineplot(
        data=df, x="selection_index", y="correct", hue="method",
        hue_order=METHODS, errorbar=None, marker="o"
    )
    g.set(
        xlabel="Selection #", ylabel="Accuracy (%)",
        title="Selection Accuracy by IMU Bearing, UWB Distance, and Sensor Fusion over All Four Experiments"
    )
    g.set_xticks(range(1, SEQ_LEN+1))
    g.set_ylim(0, 120)  # Keep 1.2 ratio but in percentage scale
    g.set_yticks(range(0, 101, 20))  # Only show ticks up to 100%
    g.legend(title="Method", frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()

# =======================
# INDIVIDUAL EXPERIMENT PLOTS
# =======================

def plot_baseline_experiment(out_png="../plots/linecharts/baseline_selections.png", out_svg="../plots/linecharts/baseline_selections.svg"):
    """Plot accuracy for Baseline experiment (Experiment 1)"""
    df_exp = selection_results[selection_results["experiment"] == 1].copy()
    df = (df_exp.groupby(["method","selection_index","file","run"], as_index=False)
               .agg(correct=("correct","mean")))
    # Convert to percentage
    df["correct"] = df["correct"] * 100
    
    # Plot
    g = sns.lineplot(
        data=df, x="selection_index", y="correct", hue="method",
        errorbar=None, marker="o"
    )
    g.set(
        xlabel="Selection #", ylabel="Accuracy (%)",
        title="Baseline Experiment: Selection Accuracy across Methods"
    )
    g.set_xticks(range(1, SEQ_LEN+1))
    g.set_ylim(0, 120)
    g.set_yticks(range(0, 101, 20))
    g.legend(title="Method", frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()

def plot_user_position_experiment(out_png="../plots/linecharts/user_position_selections.png", out_svg="../plots/linecharts/user_position_selections.svg"):
    """Plot accuracy for User Position experiment (Experiment 2)"""
    df_exp = selection_results[selection_results["experiment"] == 2].copy()
    df = (df_exp.groupby(["method","selection_index","file","run"], as_index=False)
               .agg(correct=("correct","mean")))
    # Convert to percentage
    df["correct"] = df["correct"] * 100
    
    # Plot with specified order
    g = sns.lineplot(
        data=df, x="selection_index", y="correct", hue="method",
        hue_order=METHODS, errorbar=None, marker="o"
    )
    g.set(
        xlabel="Selection #", ylabel="Accuracy (%)",
        title="User Position Experiment: Selection Accuracy with User Proximity to DC0F Anchor"
    )
    g.set_xticks(range(1, SEQ_LEN+1))
    g.set_ylim(0, 120)
    g.set_yticks(range(0, 101, 20))
    g.legend(title="Method", frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()

def plot_obstruction_experiment(out_png="../plots/linecharts/obstruction_selections.png", out_svg="../plots/linecharts/obstruction_selections.svg"):
    """Plot accuracy for Obstruction experiment (Experiment 3)"""
    df_exp = selection_results[selection_results["experiment"] == 3].copy()
    df = (df_exp.groupby(["method","selection_index","file","run"], as_index=False)
               .agg(correct=("correct","mean")))
    # Convert to percentage
    df["correct"] = df["correct"] * 100
    
    # Plot with specified order
    g = sns.lineplot(
        data=df, x="selection_index", y="correct", hue="method",
        hue_order=METHODS, errorbar=None, marker="o"
    )
    g.set(
        xlabel="Selection #", ylabel="Accuracy (%)",
        title="Obstruction Experiment: Selection Accuracy with DC0F Anchor Occluded"
    )
    g.set_xticks(range(1, SEQ_LEN+1))
    g.set_ylim(0, 120)
    g.set_yticks(range(0, 101, 20))
    g.legend(title="Method", frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()

def plot_anchor_spacing_experiment(out_png="../plots/linecharts/anchor_spacing_selections.png", out_svg="../plots/linecharts/anchor_spacing_selections.svg"):
    """Plot accuracy for Anchor Spacing experiment (Experiment 4) with separate legends for methods and line styles"""
    df_exp = selection_results[selection_results["experiment"] == 4].copy()
    df = (df_exp.groupby(["method","selection_index","file","run","exp4_spacing"], as_index=False)
               .agg(correct=("correct","mean")))
    # Convert to percentage
    df["correct"] = df["correct"] * 100
    
    # Define line styles and colors for each spacing and method
    line_styles = {"2 m": "-", "1 m": "--", "0.5 m": ":"}
    colors = {"IMU Bearing": "C0", "UWB Distance": "C1", "Sensor Fusion": "C2"}
    
    # Plot with different line styles and colors
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Store handles for legends
    method_handles = []
    spacing_handles = []
    
    for method in METHODS:
        for spacing in ["2 m", "1 m", "0.5 m"]:
            df_subset = df[(df["method"] == method) & (df["exp4_spacing"] == spacing)]
            if not df_subset.empty:
                line = ax.plot(df_subset["selection_index"], df_subset["correct"], 
                              linestyle=line_styles[spacing], marker="o", 
                              color=colors[method], label=f"{method}_{spacing}")[0]
    
    # Create dummy lines for separate legends
    # Method legend (colors)
    for method in METHODS:
        method_handles.append(plt.Line2D([0], [0], color=colors[method], label=method, marker='o'))
    
    # Spacing legend (line styles)
    for spacing, style in line_styles.items():
        spacing_handles.append(plt.Line2D([0], [0], color='black', linestyle=style, label=spacing))
    
    ax.set_xlabel("Selection #")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Anchor Spacing Experiment: Selection Accuracy at Different Anchor Spacings")
    ax.set_xticks(range(1, SEQ_LEN+1))
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, alpha=0.3)
    
    # Create two separate legends
    method_legend = ax.legend(handles=method_handles, title="Method", 
                             loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    spacing_legend = ax.legend(handles=spacing_handles, title="Anchor Spacing", 
                              loc='upper left', bbox_to_anchor=(1.02, 0.7), frameon=True)
    
    # Add the first legend back (matplotlib removes it when creating the second)
    ax.add_artist(method_legend)
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()

# =======================
# ADVANCED ANALYSIS PLOTS
# =======================

def plot_confusion_matrices():
    """Create confusion matrices for each experiment and method with experimental highlights"""
    os.makedirs("../plots/confusion_matrices", exist_ok=True)
    
    # Get baseline data for comparison
    baseline_data = selection_results[selection_results["experiment"] == 1]
    
    experiments = [1, 2, 3, 4]
    exp_names = ["Baseline", "User Position", "Obstruction", "Anchor Spacing"]
    
    for exp, exp_name in zip(experiments, exp_names):
        if exp == 4:
            # Handle spacing experiment separately - highlight 5C19 ↔ 96BB confusion
            spacings = ["2 m", "1 m", "0.5 m"]
            fig, axes = plt.subplots(len(METHODS), len(spacings), figsize=(15, 12))
            fig.suptitle(f"Confusion Matrices - {exp_name} Experiment\n(Red highlights: 5C19 ↔ 96BB confusion due to close spacing)", fontsize=14)
            
            for i, method in enumerate(METHODS):
                for j, spacing in enumerate(spacings):
                    df_sub = selection_results[
                        (selection_results["experiment"] == exp) & 
                        (selection_results["method"] == method) &
                        (selection_results["exp4_spacing"] == spacing)
                    ]
                    
                    if not df_sub.empty:
                        cm = confusion_matrix_from_df(df_sub)
                        ax = axes[i, j] if len(METHODS) > 1 else axes[j]
                        
                        # Create custom colormap for highlighting
                        import matplotlib.colors as mcolors
                        
                        # Create base heatmap
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                                  xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                                  cbar=False)
                        
                        # Find indices for 5C19 and 96BB
                        idx_5C19 = LABELS.index("5C19")
                        idx_96BB = LABELS.index("96BB")
                        
                        # Always highlight 5C19 → 96BB confusion (red border)
                        rect1 = plt.Rectangle((idx_96BB, idx_5C19), 1, 1, 
                                            fill=False, edgecolor='red', linewidth=3)
                        ax.add_patch(rect1)
                        
                        # Always highlight 96BB → 5C19 confusion (red border)
                        rect2 = plt.Rectangle((idx_5C19, idx_96BB), 1, 1, 
                                            fill=False, edgecolor='red', linewidth=3)
                        ax.add_patch(rect2)
                        
                        ax.set_title(f"{method} - {spacing}")
                        ax.set_ylabel("True" if j == 0 else "")
                        ax.set_xlabel("Predicted" if i == len(METHODS)-1 else "")
                        
        elif exp == 3:
            # Obstruction experiment - keep title but no highlighting
            fig, axes = plt.subplots(1, len(METHODS), figsize=(15, 4))
            fig.suptitle(f"Confusion Matrices - {exp_name} Experiment\n(DC0F anchor was occluded)", fontsize=14)
            
            for i, method in enumerate(METHODS):
                df_sub = selection_results[
                    (selection_results["experiment"] == exp) & 
                    (selection_results["method"] == method)
                ]
                
                if not df_sub.empty:
                    cm = confusion_matrix_from_df(df_sub)
                    ax = axes[i] if len(METHODS) > 1 else axes
                    
                    # Create base heatmap without highlighting
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                              xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                              cbar=False)
                    
                    ax.set_title(f"{method}")
                    ax.set_ylabel("True" if i == 0 else "")
                    ax.set_xlabel("Predicted")
        else:
            # Regular experiments (Baseline, User Position)
            fig, axes = plt.subplots(1, len(METHODS), figsize=(15, 4))
            if exp == 2:
                # User Position experiment - specify which anchor user was close to
                fig.suptitle(f"Confusion Matrices - {exp_name} Experiment\n(User positioned ~75cm away from DC0F anchor)", fontsize=14)
            else:
                fig.suptitle(f"Confusion Matrices - {exp_name} Experiment", fontsize=14)
            
            for i, method in enumerate(METHODS):
                df_sub = selection_results[
                    (selection_results["experiment"] == exp) & 
                    (selection_results["method"] == method)
                ]
                
                if not df_sub.empty:
                    cm = confusion_matrix_from_df(df_sub)
                    ax = axes[i] if len(METHODS) > 1 else axes
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                              xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                              cbar=False)
                    ax.set_title(f"{method}")
                    ax.set_ylabel("True" if i == 0 else "")
                    ax.set_xlabel("Predicted")
        
        plt.tight_layout()
        safe_name = exp_name.lower().replace(" ", "_")
        plt.savefig(f"../plots/confusion_matrices/confusion_matrix_{safe_name}.png", 
                   bbox_inches="tight", dpi=300)
        plt.savefig(f"../plots/confusion_matrices/confusion_matrix_{safe_name}.svg", 
                   bbox_inches="tight")
        plt.close()

def plot_baseline_comparison():
    """Show over/under-performance compared to baseline"""
    os.makedirs("../plots/comparisons", exist_ok=True)
    
    # Calculate baseline means for each method
    baseline_acc = accuracy_metrics[accuracy_metrics["experiment"] == 1].groupby("method")["accuracy"].mean()
    
    # Calculate differences for other experiments in specific order
    comparison_data = []
    
    # Define the order: User Position, Obstruction, then Spacing (2m, 1m, 0.5m)
    experiment_order = [
        (2, "User Position"),
        (3, "Obstruction"),
        (4, "Spacing 2 m", "2 m"),
        (4, "Spacing 1 m", "1 m"), 
        (4, "Spacing 0.5 m", "0.5 m")
    ]
    
    for item in experiment_order:
        if len(item) == 2:  # Regular experiments (2, 3)
            exp, exp_name = item
            exp_data = accuracy_metrics[accuracy_metrics["experiment"] == exp]
            for method in METHODS:
                method_data = exp_data[exp_data["method"] == method]
                if not method_data.empty:
                    mean_acc = method_data["accuracy"].mean()
                    baseline_mean = baseline_acc[method]
                    diff = mean_acc - baseline_mean
                    comparison_data.append({
                        "experiment": exp_name,
                        "method": method,
                        "accuracy_diff": diff * 100,
                        "baseline_acc": baseline_mean * 100,
                        "current_acc": mean_acc * 100
                    })
        else:  # Spacing experiments (4)
            exp, exp_name, spacing = item
            exp_data = accuracy_metrics[accuracy_metrics["experiment"] == exp]
            spacing_data = exp_data[exp_data["spacing"] == spacing]
            for method in METHODS:
                method_data = spacing_data[spacing_data["method"] == method]
                if not method_data.empty:
                    mean_acc = method_data["accuracy"].mean()
                    baseline_mean = baseline_acc[method]
                    diff = mean_acc - baseline_mean
                    comparison_data.append({
                        "experiment": exp_name,
                        "method": method,
                        "accuracy_diff": diff * 100,
                        "baseline_acc": baseline_mean * 100,
                        "current_acc": mean_acc * 100
                    })
    
    df_comp = pd.DataFrame(comparison_data)
    
    # Create the comparison plot
    plt.figure(figsize=(14, 8))
    
    # Create a diverging bar plot
    colors = {"IMU Bearing": "C0", "UWB Distance": "C1", "Sensor Fusion": "C2"}
    
    # Pivot for easier plotting and ensure correct order
    pivot_df = df_comp.pivot(index="experiment", columns="method", values="accuracy_diff")
    # Reorder columns to match METHODS order
    pivot_df = pivot_df.reindex(columns=METHODS)
    
    # Ensure row order matches our experiment order
    experiment_row_order = ["User Position", "Obstruction", "Spacing 2 m", "Spacing 1 m", "Spacing 0.5 m"]
    pivot_df = pivot_df.reindex(index=experiment_row_order)
    
    ax = pivot_df.plot(kind="bar", figsize=(14, 8), color=[colors[method] for method in METHODS])
    
    # Add horizontal line at 0 (baseline)
    plt.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.8)
    
    # Formatting
    plt.title("Performance Difference from Baseline\n(Positive = Better than Baseline, Negative = Worse than Baseline)", fontsize=12)
    plt.xlabel("Experimental Condition", fontsize=11)
    plt.ylabel("Accuracy Difference (Percentage Points)", fontsize=11)
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("../plots/comparisons/baseline_comparison.png", bbox_inches="tight", dpi=300)
    plt.savefig("../plots/comparisons/baseline_comparison.svg", bbox_inches="tight")
    plt.close()

def plot_fusion_disagreement_analysis():
    """Analyze fusion accuracy when IMU and UWB methods disagree - the critical decision scenarios"""
    os.makedirs("../plots/agreement", exist_ok=True)
    
    # Calculate agreement rates in specific order
    agreement_data = []
    
    # Define the order: Baseline, User Position, Obstruction, then Spacing (2m, 1m, 0.5m)
    experiment_order = [
        (1, "Baseline"),
        (2, "User Position"),
        (3, "Obstruction"),
        (4, "Spacing 2 m", "2 m"),
        (4, "Spacing 1 m", "1 m"), 
        (4, "Spacing 0.5 m", "0.5 m")
    ]
    
    for item in experiment_order:
        if len(item) == 2:  # Regular experiments (1, 2, 3)
            exp, exp_name = item
            df_sub = selection_results[selection_results["experiment"] == exp]
            if not df_sub.empty:
                imu_preds = df_sub[df_sub["method"] == "IMU Bearing"]["pred"].values
                uwb_preds = df_sub[df_sub["method"] == "UWB Distance"]["pred"].values
                fusion_preds = df_sub[df_sub["method"] == "Sensor Fusion"]["pred"].values
                true_labels = df_sub[df_sub["method"] == "IMU Bearing"]["true"].values
                
                if len(imu_preds) > 0 and len(uwb_preds) > 0:
                    agreements = (imu_preds == uwb_preds)
                    agreement_rate = agreements.mean()
                    
                    agree_mask = agreements
                    disagree_mask = ~agreements
                    
                    if agree_mask.sum() > 0:
                        agree_acc = (fusion_preds[agree_mask] == true_labels[agree_mask]).mean()
                    else:
                        agree_acc = np.nan
                        
                    if disagree_mask.sum() > 0:
                        disagree_acc = (fusion_preds[disagree_mask] == true_labels[disagree_mask]).mean()
                    else:
                        disagree_acc = np.nan
                    
                    agreement_data.append({
                        "experiment": exp_name,
                        "agreement_rate": agreement_rate * 100,
                        "accuracy_when_agree": agree_acc * 100 if not np.isnan(agree_acc) else np.nan,
                        "accuracy_when_disagree": disagree_acc * 100 if not np.isnan(disagree_acc) else np.nan
                    })
        else:  # Spacing experiments (4)
            exp, exp_name, spacing = item
            df_sub = selection_results[
                (selection_results["experiment"] == exp) &
                (selection_results["exp4_spacing"] == spacing)
            ]
            if not df_sub.empty:
                imu_preds = df_sub[df_sub["method"] == "IMU Bearing"]["pred"].values
                uwb_preds = df_sub[df_sub["method"] == "UWB Distance"]["pred"].values
                fusion_preds = df_sub[df_sub["method"] == "Sensor Fusion"]["pred"].values
                true_labels = df_sub[df_sub["method"] == "IMU Bearing"]["true"].values
                
                if len(imu_preds) > 0 and len(uwb_preds) > 0:
                    agreements = (imu_preds == uwb_preds)
                    agreement_rate = agreements.mean()
                    
                    agree_mask = agreements
                    disagree_mask = ~agreements
                    
                    if agree_mask.sum() > 0:
                        agree_acc = (fusion_preds[agree_mask] == true_labels[agree_mask]).mean()
                    else:
                        agree_acc = np.nan
                        
                    if disagree_mask.sum() > 0:
                        disagree_acc = (fusion_preds[disagree_mask] == true_labels[disagree_mask]).mean()
                    else:
                        disagree_acc = np.nan
                    
                    agreement_data.append({
                        "experiment": exp_name,
                        "agreement_rate": agreement_rate * 100,
                        "accuracy_when_agree": agree_acc * 100 if not np.isnan(agree_acc) else np.nan,
                        "accuracy_when_disagree": disagree_acc * 100 if not np.isnan(disagree_acc) else np.nan
                    })
    
    df_agreement = pd.DataFrame(agreement_data)
    
    # Ensure the DataFrame maintains the correct order
    experiment_order_for_display = ["Baseline", "User Position", "Obstruction", "Spacing 2 m", "Spacing 1 m", "Spacing 0.5 m"]
    df_agreement['experiment'] = pd.Categorical(df_agreement['experiment'], categories=experiment_order_for_display, ordered=True)
    df_agreement = df_agreement.sort_values('experiment')
    
    # Create the two-panel analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Agreement rates
    bars1 = ax1.bar(df_agreement["experiment"], df_agreement["agreement_rate"], 
                   color="steelblue", alpha=0.7)
    ax1.set_title("IMU-UWB Method Agreement Rates", fontsize=12)
    ax1.set_ylabel("Agreement Rate (%)", fontsize=11)
    ax1.set_xticklabels(df_agreement["experiment"], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, df_agreement["agreement_rate"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Accuracy only when methods disagree (remove green bars and legend)
    x = np.arange(len(df_agreement))
    
    bars2 = ax2.bar(x, df_agreement["accuracy_when_disagree"], 
                   color="darkred", alpha=0.7)
    
    ax2.set_title("Fusion Accuracy When Methods Disagree", fontsize=12)
    ax2.set_ylabel("Fusion Accuracy (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_agreement["experiment"], rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, df_agreement["accuracy_when_disagree"]):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("../plots/agreement/method_disagreement_analysis.png", bbox_inches="tight", dpi=300)
    plt.savefig("../plots/agreement/method_disagreement_analysis.svg", bbox_inches="tight")
    plt.close()

def plot_spacing_detailed_analysis():
    """Detailed analysis of spacing experiment showing proximity effects"""
    os.makedirs("../plots/spacing", exist_ok=True)
    
    spacing_data = selection_results[selection_results["experiment"] == 4]
    
    if spacing_data.empty:
        print("No spacing experiment data found")
        return
    
    # 1. Single bar plot for proximity confusion analysis
    spacings = ["2 m", "1 m", "0.5 m"]
    spacing_colors = {"2 m": "#2E8B57", "1 m": "#FF8C00", "0.5 m": "#DC143C"}  # Green, Orange, Red
    
    confusion_data = []
    
    for spacing in spacings:
        df_sub = spacing_data[spacing_data["exp4_spacing"] == spacing]
        
        for method in METHODS:
            method_data = df_sub[df_sub["method"] == method]
            if not method_data.empty:
                conf_rate = pair_confusion_rate(method_data, "5C19", "96BB")
                if not np.isnan(conf_rate):
                    confusion_data.append({
                        "method": method,
                        "spacing": spacing,
                        "confusion_rate": conf_rate * 100
                    })
    
    if confusion_data:
        df_confusion = pd.DataFrame(confusion_data)
        
        # Create single bar plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Group by method and plot bars for each spacing
        x_positions = np.arange(len(METHODS))
        bar_width = 0.25
        
        for i, spacing in enumerate(spacings):
            spacing_subset = df_confusion[df_confusion["spacing"] == spacing]
            confusion_values = []
            
            for method in METHODS:
                method_data = spacing_subset[spacing_subset["method"] == method]
                if not method_data.empty:
                    confusion_values.append(method_data["confusion_rate"].iloc[0])
                else:
                    confusion_values.append(0)
            
            bars = ax.bar(x_positions + i * bar_width, confusion_values, 
                         bar_width, label=f"{spacing} spacing", 
                         color=spacing_colors[spacing], alpha=0.8)
            
            # Add value labels
            for j, (bar, val) in enumerate(zip(bars, confusion_values)):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel("Confusion Rate (%)", fontsize=12)
        ax.set_title("Proximity Effect: Confusion Between Close Anchors (5C19, 96BB) Across Spacing Distances", fontsize=14)
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(METHODS)
        ax.legend(title="Anchor Spacing", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("../plots/spacing/proximity_confusion_analysis.png", bbox_inches="tight", dpi=300)
        plt.savefig("../plots/spacing/proximity_confusion_analysis.svg", bbox_inches="tight")
        plt.close()
    
    # 2. Accuracy degradation with spacing
    spacing_accuracy = []
    
    for spacing in spacings:
        for method in METHODS:
            df_sub = spacing_data[
                (spacing_data["exp4_spacing"] == spacing) & 
                (spacing_data["method"] == method)
            ]
            if not df_sub.empty:
                acc = df_sub["correct"].mean() * 100
                spacing_accuracy.append({
                    "spacing": spacing,
                    "method": method,
                    "accuracy": acc,
                    "spacing_numeric": float(spacing.split()[0])  # For ordering
                })
    
    df_spacing_acc = pd.DataFrame(spacing_accuracy)
    
    plt.figure(figsize=(10, 6))
    colors = {"IMU Bearing": "C0", "UWB Distance": "C1", "Sensor Fusion": "C2"}
    for method in METHODS:
        method_data = df_spacing_acc[df_spacing_acc["method"] == method]
        if not method_data.empty:
            plt.plot(method_data["spacing_numeric"], method_data["accuracy"], 
                    marker="o", label=method, linewidth=2, markersize=8, color=colors[method])
    
    plt.xlabel("Anchor Spacing (meters)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.title("Target Selection Accuracy vs Anchor Spacing", fontsize=12)
    plt.legend(title="Method")
    plt.grid(True, alpha=0.3)
    plt.xticks([0.5, 1.0, 2.0])
    
    # Add annotations for key insights
    plt.annotate("Close spacing\nchallenges", xy=(0.5, 50), xytext=(0.7, 30),
                arrowprops=dict(arrowstyle="->", alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    plt.savefig("../plots/spacing/accuracy_vs_spacing.png", bbox_inches="tight", dpi=300)
    plt.savefig("../plots/spacing/accuracy_vs_spacing.svg", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("Files used:", *sorted(files), sep="\n- ")
    print("\nPer-file accuracies (mean over 20 selections):")
    print(accuracy_metrics.sort_values(["experiment","method","file"]).to_string(index=False))

    # Create output directories
    os.makedirs("../plots/linecharts", exist_ok=True)
    os.makedirs("../plots/confusion_matrices", exist_ok=True)
    os.makedirs("../plots/comparisons", exist_ok=True)
    os.makedirs("../plots/agreement", exist_ok=True)
    os.makedirs("../plots/spacing", exist_ok=True)

    # Generate all plots
    print("\n=== Generating Basic Line Charts ===")
    fig1_accuracy_over_selections()
    plot_baseline_experiment()
    plot_user_position_experiment()
    plot_obstruction_experiment()
    plot_anchor_spacing_experiment()

    print("\n=== Generating Advanced Analysis Plots ===")
    plot_confusion_matrices()
    plot_baseline_comparison()
    plot_fusion_disagreement_analysis()
    plot_spacing_detailed_analysis()

    print("\n=== Plot Generation Complete ===")
    print("Saved line charts in ../plots/linecharts/:")
    print("- Overall: fig1_selections.(png,svg)")
    print("- Baseline: baseline_selections.(png,svg)")
    print("- User Position: user_position_selections.(png,svg)")
    print("- Obstruction: obstruction_selections.(png,svg)")
    print("- Anchor Spacing: anchor_spacing_selections.(png,svg)")
    
    print("\nSaved advanced analysis plots:")
    print("- Confusion matrices: ../plots/confusion_matrices/")
    print("- Baseline comparisons: ../plots/comparisons/baseline_comparison.(png,svg)")
    print("- Method disagreement analysis: ../plots/agreement/method_disagreement_analysis.(png,svg)")
    print("- Spacing analysis: ../plots/spacing/")
    print("  * proximity_confusion_analysis.(png,svg)")
    print("  * accuracy_vs_spacing.(png,svg)")
