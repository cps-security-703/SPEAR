

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


INPUT_FILE = "top_rl_actions_for_simulation.json"
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)


RAG_COLOR = "#2196F3"
NON_RAG_COLOR = "#FF5722"
COMPONENT_COLORS = {
    "cve_score": "#4CAF50",
    "mitre_score": "#9C27B0",
    "rl_action_score": "#FF9800",
    "protocol_score": "#00BCD4",
    "context_score": "#2196F3",
    "structure_score": "#607D8B",
    "hallucination_penalty": "#F44336",
}
COMPONENT_LABELS = {
    "cve_score": "CVE Score",
    "mitre_score": "MITRE Score",
    "rl_action_score": "RL Action Specificity",
    "protocol_score": "Protocol Specificity",
    "context_score": "Context Usage",
    "structure_score": "Structured Format",
    "hallucination_penalty": "Hallucination Penalty",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def plot_total_confidence(data, save=True):
    results = data["all_results"]
    query_ids = [r["query_id"].replace("_", "\n", 1) for r in results]
    rag_scores = [r["rag"]["confidence"]["total_confidence"] for r in results]
    non_rag_scores = [r["non_rag"]["confidence"]["total_confidence"] for r in results]

    x = np.arange(len(query_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    bars_rag = ax.bar(x - width / 2, rag_scores, width, label="RAG", color=RAG_COLOR, edgecolor="white", linewidth=1)
    bars_non = ax.bar(x + width / 2, non_rag_scores, width, label="Non-RAG", color=NON_RAG_COLOR, edgecolor="white", linewidth=1)


    for bar in bars_rag:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=14, fontweight="bold", color=RAG_COLOR)
    for bar in bars_non:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=14, fontweight="bold", color=NON_RAG_COLOR)

    ax.set_xlabel("Query", fontsize=18)
    ax.set_ylabel("Confidence Score", fontsize=18)

    ax.set_xticks(x)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xticklabels(query_ids, rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0, max(max(rag_scores), max(non_rag_scores)) + 12)


    rag_avg_line = ax.axhline(y=data["summary"]["avg_confidence"]["rag"], color=RAG_COLOR, linestyle="--", alpha=0.5, linewidth=1,
                               label=f'RAG avg: {data["summary"]["avg_confidence"]["rag"]:.1f}')
    non_rag_avg_line = ax.axhline(y=data["summary"]["avg_confidence"]["non_rag"], color=NON_RAG_COLOR, linestyle="--", alpha=0.5, linewidth=1,
                                   label=f'Non-RAG avg: {data["summary"]["avg_confidence"]["non_rag"]:.1f}')

    ax.legend(loc=  "lower right", fontsize=12)

    plt.tight_layout()
    plt.grid(True)
    if save:
        fig.savefig(OUTPUT_DIR / "1_total_confidence_comparison.pdf")
    plt.show()


def plot_confidence_advantage(data, save=True):
    results = data["all_results"]
    query_ids = [r["query_id"] for r in results]
    advantages = [
        r["rag"]["confidence"]["total_confidence"] - r["non_rag"]["confidence"]["total_confidence"]
        for r in results
    ]


    sorted_pairs = sorted(zip(query_ids, advantages), key=lambda x: x[1])
    query_ids_sorted, advantages_sorted = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [RAG_COLOR if a > 0 else NON_RAG_COLOR for a in advantages_sorted]
    y = np.arange(len(query_ids_sorted))

    ax.hlines(y, 0, advantages_sorted, colors=colors, linewidth=2.5, alpha=0.8)
    ax.scatter(advantages_sorted, y, color=colors, s=80, zorder=5, edgecolors="white", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(query_ids_sorted, fontsize=15)
    ax.set_xlabel("Confidence Advantage (RAG − Non-RAG)", fontsize=15, fontweight="bold")

    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)

    avg_adv = data["summary"]["avg_confidence"]["advantage"]
    ax.axvline(x=avg_adv, color=RAG_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(avg_adv + 1, len(query_ids_sorted) - 1, f"Avg: +{avg_adv:.1f}", fontsize=9, color=RAG_COLOR)

    rag_patch = mpatches.Patch(color=RAG_COLOR, label="RAG wins")
    non_rag_patch = mpatches.Patch(color=NON_RAG_COLOR, label="Non-RAG wins")
    ax.legend(handles=[rag_patch, non_rag_patch], loc="lower right", fontsize=18)

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "2_confidence_advantage.pdf", bbox_inches="tight")
    plt.show()


def plot_component_breakdown(data, save=True):
    results = data["all_results"]
    components = ["cve_score", "mitre_score", "rl_action_score", "protocol_score", "context_score", "structure_score"]


    rag_avg = {c: np.mean([r["rag"]["confidence"]["components"][c] for r in results]) for c in components}
    non_rag_avg = {c: np.mean([r["non_rag"]["confidence"]["components"][c] for r in results]) for c in components}
    rag_penalty = np.mean([r["rag"]["confidence"]["components"]["hallucination_penalty"] for r in results])
    non_rag_penalty = np.mean([r["non_rag"]["confidence"]["components"]["hallucination_penalty"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))


    categories = ["RAG", "Non-RAG"]
    bottoms = [0, 0]
    for comp in components:
        vals = [rag_avg[comp], non_rag_avg[comp]]
        axes[0].bar(categories, vals, bottom=bottoms, label=COMPONENT_LABELS[comp],
                    color=COMPONENT_COLORS[comp], edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 1.5:
                axes[0].text(i, b + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        bottoms = [b + v for b, v in zip(bottoms, vals)]


    penalty_vals = [rag_penalty, non_rag_penalty]
    axes[0].bar(categories, penalty_vals, bottom=0, label=COMPONENT_LABELS["hallucination_penalty"],
                color=COMPONENT_COLORS["hallucination_penalty"], edgecolor="white", linewidth=0.5, alpha=0.7)
    for i, v in enumerate(penalty_vals):
        if v < -0.5:
            axes[0].text(i, v / 2, f"{v:.1f}", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    axes[0].set_ylabel("Score Points")

    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axhline(y=0, color="black", linewidth=0.5)


    query_ids = [r["query_id"] for r in results]
    all_components = components + ["hallucination_penalty"]
    matrix = np.array([
        [r["rag"]["confidence"]["components"][c] for c in all_components]
        for r in results
    ])

    im = axes[1].imshow(matrix.T, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    axes[1].set_xticks(range(len(query_ids)))
    axes[1].set_xticklabels([q.split("_", 1)[0] for q in query_ids], rotation=45, ha="right", fontsize=18)
    axes[1].set_yticks(range(len(all_components)))
    axes[1].set_yticklabels([COMPONENT_LABELS[c] for c in all_components], fontsize=18)


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > 10 else "black"
            axes[1].text(i, j, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=axes[1], shrink=0.8, label="Score")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "3_component_breakdown.pdf", bbox_inches="tight")
    plt.show()


def plot_radar_comparison(data, save=True):
    results = data["all_results"]
    components = ["cve_score", "mitre_score", "rl_action_score", "protocol_score", "context_score", "structure_score"]
    max_scores = {"cve_score": 25, "mitre_score": 20, "rl_action_score": 15, "protocol_score": 10, "context_score": 20, "structure_score": 5}


    rag_vals = [np.mean([r["rag"]["confidence"]["components"][c] for r in results]) / max_scores[c] * 100 for c in components]
    non_rag_vals = [np.mean([r["non_rag"]["confidence"]["components"][c] for r in results]) / max_scores[c] * 100 for c in components]

    labels = [COMPONENT_LABELS[c] for c in components]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    rag_vals += rag_vals[:1]
    non_rag_vals += non_rag_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, rag_vals, color=RAG_COLOR, alpha=0.15)
    ax.plot(angles, rag_vals, color=RAG_COLOR, linewidth=2, label="RAG")
    ax.fill(angles, non_rag_vals, color=NON_RAG_COLOR, alpha=0.15)
    ax.plot(angles, non_rag_vals, color=NON_RAG_COLOR, linewidth=2, label="Non-RAG")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=15, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=18, color="gray")

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.grid(True)
    if save:
        fig.savefig(OUTPUT_DIR / "4_radar_comparison.pdf", bbox_inches="tight")
    plt.show()


def plot_hallucination_analysis(data, save=True):
    results = data["all_results"]
    query_ids = [r["query_id"] for r in results]

    rag_verified = [len(r["rag"]["confidence"].get("verified_cves", [])) for r in results]
    rag_unverified = [len(r["rag"]["confidence"].get("unverified_cves", [])) for r in results]
    non_rag_verified = [len(r["non_rag"]["confidence"].get("verified_cves", [])) for r in results]
    non_rag_unverified = [len(r["non_rag"]["confidence"].get("unverified_cves", [])) for r in results]

    x = np.arange(len(query_ids))
    width = 0.75
    total_rag_halluc = sum(rag_unverified)
    total_non_rag_halluc = sum(non_rag_unverified)


    fig_a, ax_a = plt.subplots(figsize=(10, 8))
    ax_a.bar(x, rag_verified, width, label="Verified CVEs", color="#4CAF50", edgecolor="white")
    ax_a.bar(x, rag_unverified, width, bottom=rag_verified, label="Unverified (Hallucinated)", color="#F44336", edgecolor="white")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([q.split("_", 1)[0] for q in query_ids], rotation=45, ha="right", fontsize=24)
    ax_a.set_ylabel("Number of CVEs", fontsize=24)


    ax_a.plot([], [], ' ', label=f"Total hallucinated: {total_rag_halluc}")
    ax_a.legend(fontsize=24, loc="upper right")
    fig_a.tight_layout()
    plt.grid(True)
    if save:
        fig_a.savefig(OUTPUT_DIR / "5a_hallucination_rag.pdf", bbox_inches="tight")
    plt.show()


    fig_b, ax_b = plt.subplots(figsize=(10, 8))
    ax_b.bar(x, non_rag_verified, width, label="Verified CVEs", color="#4CAF50", edgecolor="white")
    ax_b.bar(x, non_rag_unverified, width, bottom=non_rag_verified, label="Unverified (Hallucinated)", color="#F44336", edgecolor="white")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([q.split("_", 1)[0] for q in query_ids], rotation=45, ha="right", fontsize=24)
    ax_b.set_ylabel("Number of CVEs", fontsize=24)


    ax_b.plot([], [], ' ', label=f"Total hallucinated: {total_non_rag_halluc}")
    ax_b.legend(fontsize=24, loc="upper right")
    fig_b.tight_layout()
    plt.grid(True)
    if save:
        fig_b.savefig(OUTPUT_DIR / "5b_hallucination_nonrag.pdf", bbox_inches="tight")
    plt.show()


def plot_top_actions(data, save=True):
    top_actions = data.get("top_rl_actions", [])
    if not top_actions:
        print("No top_rl_actions found in data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))


    action_labels = [f"Action {i+1}\n({a['query_id']})" for i, a in enumerate(top_actions)]
    confidences = [a["confidence_score"] for a in top_actions]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_actions)))

    bars = axes[0].barh(action_labels, confidences, color=colors, edgecolor="white", linewidth=0.5)
    for bar, conf in zip(bars, confidences):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{conf:.0f}", va="center", fontsize=18, fontweight="bold")
    axes[0].set_xlabel("Confidence Score", fontsize=18)

    axes[0].set_xlim(0, max(confidences) + 10)
    axes[0].invert_yaxis()


    stride_cats = {}
    protocol_cats = {}
    for a in top_actions:
        for s in a.get("stride_categories", []):
            stride_cats[s] = stride_cats.get(s, 0) + 1
        for p in a.get("protocols", []):
            protocol_cats[p] = protocol_cats.get(p, 0) + 1


    if stride_cats:
        stride_colors = plt.cm.Set2(np.linspace(0, 1, len(stride_cats)))
        wedges, texts, autotexts = axes[1].pie(
            stride_cats.values(), labels=stride_cats.keys(), autopct="%1.0f%%",
            colors=stride_colors, startangle=90, pctdistance=0.85, textprops={"fontsize": 8}
        )
        centre_circle = plt.Circle((0, 0), 0.55, fc="white")
        axes[1].add_artist(centre_circle)


        proto_text = "Protocols:\n" + "\n".join(f"  {p} ({c})" for p, c in protocol_cats.items())
        axes[1].text(0, 0, proto_text, ha="center", va="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "6_top_actions_summary.pdf", bbox_inches="tight")
    plt.show()


def plot_summary_dashboard(data, save=True):
    summary = data["summary"]
    results = data["all_results"]

    fig = plt.figure(figsize=(16, 10))


    ax_cards = fig.add_axes([0.05, 0.85, 0.9, 0.1])
    ax_cards.axis("off")
    metrics = [
        ("RAG Wins", f"{summary['rag_wins']}/{data['total_queries']}", RAG_COLOR),
        ("Avg RAG Score", f"{summary['avg_confidence']['rag']:.1f}", RAG_COLOR),
        ("Avg Non-RAG Score", f"{summary['avg_confidence']['non_rag']:.1f}", NON_RAG_COLOR),
        ("RAG Advantage", f"+{summary['avg_confidence']['advantage']:.1f}", "#4CAF50"),
        ("Top Action Range", f"{summary['top_actions_confidence_range']['lowest']:.0f}–{summary['top_actions_confidence_range']['highest']:.0f}", "#9C27B0"),
    ]
    for i, (label, value, color) in enumerate(metrics):
        x_pos = 0.1 + i * 0.18
        ax_cards.text(x_pos, 0.7, value, fontsize=20, fontweight="bold", color=color, ha="center", va="center")
        ax_cards.text(x_pos, 0.15, label, fontsize=10, color="gray", ha="center", va="center")


    ax1 = fig.add_axes([0.06, 0.08, 0.55, 0.7])
    query_ids = [r["query_id"].replace("_", "\n", 1) for r in results]
    rag_scores = [r["rag"]["confidence"]["total_confidence"] for r in results]
    non_rag_scores = [r["non_rag"]["confidence"]["total_confidence"] for r in results]
    x = np.arange(len(query_ids))
    width = 0.35
    ax1.bar(x - width / 2, rag_scores, width, label="RAG", color=RAG_COLOR, edgecolor="white")
    ax1.bar(x + width / 2, non_rag_scores, width, label="Non-RAG", color=NON_RAG_COLOR, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(query_ids, rotation=45, ha="right", fontsize=18)
    ax1.set_ylabel("Confidence Score", fontsize=18)

    ax1.legend(fontsize=18)


    components = ["cve_score", "mitre_score", "rl_action_score", "protocol_score", "context_score", "structure_score"]
    max_scores = {"cve_score": 25, "mitre_score": 20, "rl_action_score": 15, "protocol_score": 10, "context_score": 20, "structure_score": 5}
    rag_vals = [np.mean([r["rag"]["confidence"]["components"][c] for r in results]) / max_scores[c] * 100 for c in components]
    non_rag_vals = [np.mean([r["non_rag"]["confidence"]["components"][c] for r in results]) / max_scores[c] * 100 for c in components]
    labels = [COMPONENT_LABELS[c] for c in components]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    rag_vals += rag_vals[:1]
    non_rag_vals += non_rag_vals[:1]
    angles += angles[:1]

    ax2 = fig.add_axes([0.62, 0.08, 0.38, 0.7], polar=True)
    ax2.fill(angles, rag_vals, color=RAG_COLOR, alpha=0.15)
    ax2.plot(angles, rag_vals, color=RAG_COLOR, linewidth=2, label="RAG")
    ax2.fill(angles, non_rag_vals, color=NON_RAG_COLOR, alpha=0.15)
    ax2.plot(angles, non_rag_vals, color=NON_RAG_COLOR, linewidth=2, label="Non-RAG")
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, fontsize=18)
    ax2.set_ylim(0, 110)
    ax2.set_yticks([25, 50, 75, 100])
    ax2.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=18, color="gray")

    ax2.legend(fontsize=18, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if save:
        fig.savefig(OUTPUT_DIR / "7_summary_dashboard.pdf", bbox_inches="tight")
    plt.show()


def main():
    print(f"Loading data from {INPUT_FILE}...")
    data = load_data(INPUT_FILE)
    print(f"  {data['total_queries']} queries, {data['actions_selected']} top actions selected")
    print(f"  RAG wins: {data['summary']['rag_wins']}/{data['total_queries']}")
    print(f"  Avg confidence — RAG: {data['summary']['avg_confidence']['rag']:.1f}, "
          f"Non-RAG: {data['summary']['avg_confidence']['non_rag']:.1f}")
    print()

    print("Generating plots...")

    print("  [1/7] Total confidence comparison...")
    plot_total_confidence(data)

    print("  [2/7] Confidence advantage lollipop...")
    plot_confidence_advantage(data)

    print("  [3/7] Component breakdown + heatmap...")
    plot_component_breakdown(data)

    print("  [4/7] Radar comparison...")
    plot_radar_comparison(data)

    print("  [5/7] Hallucination analysis...")
    plot_hallucination_analysis(data)

    print("  [6/7] Top actions summary...")
    plot_top_actions(data)

    print("  [7/7] Summary dashboard...")
    plot_summary_dashboard(data)

    print(f"\nAll plots saved to {OUTPUT_DIR.resolve()}/")
    print("Done!")


if __name__ == "__main__":
    main()
