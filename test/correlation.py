"""
Étape 7 — Corrélation entre le score BERT et le gold standard linguistique.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, kendalltau
from pathlib import Path


def fusionner_scores(df_bert: pd.DataFrame, df_gold: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les deux DataFrames sur l'index (mot)."""
    df = df_bert[["score_polysemie", "mean_sim", "spread"]].join(
        df_gold[["n_sens_manuel", "score_gold"]], how="inner"
    )
    df.dropna(subset=["score_polysemie", "score_gold"], inplace=True)
    return df


def calculer_correlations(df: pd.DataFrame) -> dict:
    """
    Calcule Pearson, Spearman et Kendall entre toutes les métriques BERT
    et le gold standard.
    """
    metriques_bert = ["score_polysemie", "mean_sim", "spread"]
    gold_cols      = ["score_gold", "n_sens_manuel"]
    resultats = {}

    print("\n Corrélations entre métriques BERT et gold standard :\n")
    print(f"{'Métrique BERT':<20} {'Gold':<18} {'Pearson r':<12} {'Spearman ρ':<12} {'Kendall τ':<12} {'p-val (S)'}")
    print("─" * 90)

    for mb in metriques_bert:
        for gc in gold_cols:
            x = df[mb].values
            y = df[gc].dropna().values
            # S'assurer que les longueurs correspondent après dropna
            communs = df[[mb, gc]].dropna()
            x, y = communs[mb].values, communs[gc].values
            if len(x) < 3:
                continue
            r_p, p_p   = pearsonr(x, y)
            r_s, p_s   = spearmanr(x, y)
            r_k, p_k   = kendalltau(x, y)
            sig = "***" if p_s < 0.001 else ("**" if p_s < 0.01 else ("*" if p_s < 0.05 else "ns"))
            key = f"{mb}_vs_{gc}"
            resultats[key] = {"pearson": r_p, "spearman": r_s, "kendall": r_k,
                               "p_spearman": p_s, "sig": sig}
            print(
                f"{mb:<20} {gc:<18} {r_p:+.4f}      {r_s:+.4f}      {r_k:+.4f}      "
                f"p={p_s:.4f} {sig}"
            )

    return resultats


def figure_scatter(df: pd.DataFrame, dossier: str = "resultats/figures"):
    """
    Scatter plot : score BERT vs score gold.
    Chaque point est un mot, avec son nom annoté.
    """
    Path(dossier).mkdir(parents=True, exist_ok=True)

    x = df["score_polysemie"]
    y = df["score_gold"]
    r_s, p_s = spearmanr(x, y)
    r_p, p_p = pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Points
    couleurs = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(df)))
    ax.scatter(x, y, s=150, c=couleurs, edgecolors="gray", linewidths=0.8, zorder=3)

    # Annotations
    for mot, row in df.iterrows():
        ax.annotate(
            f"  {mot}", (row["score_polysemie"], row["score_gold"]),
            fontsize=10, va="center", ha="left"
        )

    # Droite de régression
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, "b--", alpha=0.5, linewidth=1.5, label="Régression linéaire")

    ax.set_xlabel("Score BERT (écart-type des sim. cosinus)", fontsize=12)
    ax.set_ylabel("Score gold standard (normalisé)", fontsize=12)
    ax.set_title(
        f"Corrélation score BERT ↔ polysémie linguistique\n"
        f"Spearman ρ = {r_s:+.3f} (p={p_s:.3f})   |   Pearson r = {r_p:+.3f}",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    chemin = Path(dossier) / "correlation_scatter.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Scatter plot sauvegardé : {chemin}")


def figure_comparaison_barres(df: pd.DataFrame, dossier: str = "resultats/figures"):
    """
    Double graphique à barres : score BERT et score gold normalisés côte à côte.
    Permet de voir d'un coup d'œil les concordances et discordances.
    """
    Path(dossier).mkdir(parents=True, exist_ok=True)

    df_trie = df.sort_values("score_gold", ascending=False)
    mots = df_trie.index.tolist()
    x = np.arange(len(mots))
    width = 0.35

    # Normaliser le score BERT pour comparaison visuelle
    s_bert = df_trie["score_polysemie"]
    s_bert_norm = (s_bert - s_bert.min()) / (s_bert.max() - s_bert.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, s_bert_norm, width, label="Score BERT (normalisé)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, df_trie["score_gold"], width, label="Gold standard (normalisé)", color="#E63946", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"« {m} »" for m in mots], rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("Score normalisé [0, 1]")
    ax.set_title("Comparaison scores BERT vs gold standard\n(barres proches = bonne correspondance)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    chemin = Path(dossier) / "comparaison_barres.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Graphique comparatif sauvegardé : {chemin}")


def figure_heatmap_correlations(df: pd.DataFrame, dossier: str = "resultats/figures"):
    """Heatmap de la matrice de corrélation (Spearman) entre toutes les métriques."""
    Path(dossier).mkdir(parents=True, exist_ok=True)
    cols = ["score_polysemie", "mean_sim", "spread", "score_gold", "n_sens_manuel"]
    df_sub = df[cols].dropna()

    corr_mat = df_sub.corr(method="spearman")
    etiquettes = ["Score std BERT", "Mean sim BERT", "Spread BERT",
                  "Gold normalisé", "Sens FR (TLFi)"]

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, ax=ax,
        xticklabels=etiquettes, yticklabels=etiquettes,
        linewidths=0.5, linecolor="white"
    )
    ax.set_title("Matrice de corrélation Spearman\nentre métriques BERT et gold standard", fontsize=12)
    plt.tight_layout()
    chemin = Path(dossier) / "heatmap_correlations.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Heatmap de corrélations sauvegardée : {chemin}")


def sauvegarder_resultats(df: pd.DataFrame, correlations: dict, dossier: str = "resultats"):
    """Sauvegarde le tableau fusionné et les corrélations."""
    Path(dossier).mkdir(exist_ok=True)
    df.to_csv(Path(dossier) / "tableau_final.csv")
    import json
    # Convertir les valeurs numpy en float pour JSON
    corr_serialisable = {k: {kk: float(vv) for kk, vv in v.items() if kk != "sig"}
                         for k, v in correlations.items()}
    with open(Path(dossier) / "correlations.json", "w", encoding="utf-8") as f:
        json.dump(corr_serialisable, f, indent=2)
    print(f"\n Résultats finaux sauvegardés dans « {dossier}/ »")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from scores import charger_scores
    from gold_manuel import charger_gold_standard

    df_bert = charger_scores("resultats")
    df_gold = charger_gold_standard("resultats")

    print(" Fusion des scores…")
    df = fusionner_scores(df_bert, df_gold)
    print(df[["score_polysemie", "score_gold"]].to_string())

    correlations = calculer_correlations(df)

    print("\n Génération des figures…")
    figure_scatter(df)
    figure_comparaison_barres(df)
    figure_heatmap_correlations(df)
    sauvegarder_resultats(df, correlations)

    print("\n Étape 7 terminée.")