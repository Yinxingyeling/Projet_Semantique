"""
Étape 5 — Visualisation 2D des embeddings par réduction de dimensionnalité.
Utilise UMAP (recommandé) avec t-SNE en fallback.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from pathlib import Path

# Palette de couleurs distinctes
PALETTE = [
    "#B83F49", "#2196F3", "#2DC653", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
    "#9C857F", "#559089", "#BB8463", "#F06FEE",
    "#AEB2B8", "#5EA2459C", "#C9CF59", "#D9B3E2"
]


def reduire_dimensions(embeddings_concat: np.ndarray, methode: str = "umap", seed: int = 42):
    """
    Réduit les embeddings en 2D.
    methode : 'umap' (recommandé) ou 'tsne'
    """
    if methode == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=seed, n_neighbors=10, min_dist=0.1)
            print("  Réduction UMAP…")
        except ImportError:
            print("  umap-learn non installé, passage à t-SNE.")
            methode = "tsne"

    if methode == "tsne":
        from sklearn.manifold import TSNE
        n = len(embeddings_concat)
        perplexite = min(30, n // 3)
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perplexite)
        print(f"  Réduction t-SNE (perplexité={perplexite})…")

    coords = reducer.fit_transform(embeddings_concat)
    return coords


def figure_par_mot(embeddings: dict, dossier_sortie: str = "resultats/figures"):
    """
    Crée une figure par mot montrant la distribution des embeddings en 2D.
    Les points sont numérotés pour identifier les phrases.
    """
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)

    for mot, arr in embeddings.items():
        n = len(arr)
        coords = reduire_dimensions(arr, methode="umap")

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=range(n), cmap="viridis", s=120, edgecolors="white", linewidths=0.8, zorder=3
        )
        for i in range(n):
            ax.annotate(str(i + 1), (coords[i, 0], coords[i, 1]),
                        fontsize=7, ha="center", va="center", color="white", fontweight="bold")

        plt.colorbar(scatter, ax=ax, label="Numéro de phrase")
        ax.set_title(f"Embeddings contextuels de « {mot} »\n(chaque point = une occurrence)", fontsize=13)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        chemin = Path(dossier_sortie) / f"umap_{mot}.png"
        plt.savefig(chemin, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Figure sauvegardée : {chemin}")


def figure_globale(embeddings: dict, dossier_sortie: str = "resultats/figures"):
    """
    Projette TOUS les embeddings sur un seul graphe 2D.
    Chaque mot a sa propre couleur.
    Permet de comparer visuellement la dispersion entre mots.
    """
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)

    mots = list(embeddings.keys())
    concat = np.vstack([embeddings[m] for m in mots])
    labels = np.concatenate([[i] * len(embeddings[m]) for i, m in enumerate(mots)])

    print("\n Projection globale (tous les mots ensemble)…")
    coords = reduire_dimensions(concat, methode="umap")

    fig, ax = plt.subplots(figsize=(12, 9))
    legendes = []
    for i, mot in enumerate(mots):
        masque = labels == i
        couleur = PALETTE[i % len(PALETTE)]
        ax.scatter(
            coords[masque, 0], coords[masque, 1],
            color=couleur, s=80, alpha=0.8,
            edgecolors="white", linewidths=0.5, label=mot, zorder=3
        )
        # Centre du cluster
        cx, cy = coords[masque, 0].mean(), coords[masque, 1].mean()
        ax.text(cx, cy, f"« {mot} »", fontsize=10, fontweight="bold",
                color=couleur, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=couleur))
        legendes.append(mpatches.Patch(color=couleur, label=mot))

    ax.legend(handles=legendes, loc="upper right", fontsize=10)
    ax.set_title("Projection UMAP — Comparaison de la dispersion par mot\n"
                 "(Une dispersion plus large = une plus grande variation d’usage)", fontsize=13)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    chemin = Path(dossier_sortie) / "umap_global.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Figure globale sauvegardée : {chemin}")


def figure_heatmaps(embeddings: dict, dossier_sortie: str = "resultats/figures"):
    """
    Matrices de similarité cosinus pour chaque mot.
    Un bloc diagonalisé → plusieurs sens distincts (polysémique).
    Une matrice uniforme → sens unique (monosémique).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    mots = list(embeddings.keys())
    n_mots = len(mots)
    cols = min(3, n_mots)
    rows = (n_mots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1) if n_mots > 1 else [axes]

    for ax, mot in zip(axes, mots):
        mat = cosine_similarity(embeddings[mot])
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"« {mot} »\nstd={np.std(mat[np.triu_indices_from(mat, k=1)]):.4f}", fontsize=11)
        ax.set_xlabel("Occurrence n°")
        ax.set_ylabel("Occurrence n°")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n_mots:]:
        ax.axis("off")

    plt.suptitle("Matrices de similarité cosinus par mot\n"
                 "(vert=similaire, rouge=différent)", fontsize=14, y=1.01)
    plt.tight_layout()
    chemin = Path(dossier_sortie) / "heatmaps_similarites.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Heatmaps sauvegardées : {chemin}")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from embedding import charger_embeddings

    print(" Chargement des embeddings…")
    embeddings = charger_embeddings("resultats")

    print("\n Génération des figures par mot…")
    figure_par_mot(embeddings)

    print("\n Génération de la figure globale…")
    figure_globale(embeddings)

    print("\n Génération des heatmaps…")
    figure_heatmaps(embeddings)

    print("\n Visualisation terminée.")
