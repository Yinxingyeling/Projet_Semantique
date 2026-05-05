"""
Étape 4 — Calcul du score de polysémie
basé sur l'écart-type des similarités cosinus 2 à 2.
"""

import json
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def similarites_cosinus_paires(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcule toutes les similarités cosinus 2 à 2 pour une matrice d'embeddings.
    Retourne un vecteur plat des C(n,2) similarités.
    """
    n = len(embeddings)
    if n < 2:
        raise ValueError("Il faut au moins 2 embeddings pour calculer les similarités.")
    # Matrice cosinus complète (n x n)
    mat = cosine_similarity(embeddings)
    # Extraire le triangle supérieur (sans la diagonale)
    indices = list(combinations(range(n), 2))
    sims = np.array([mat[i, j] for i, j in indices])
    return sims


def score_polysemie(embeddings: np.ndarray) -> dict:
    """
    Calcule le score de polysémie et plusieurs métriques associées.

    Score principal : écart-type des similarités cosinus 2 à 2.
    Un fort écart-type indique que certaines paires partagent le même sens
    (sim élevée) et d'autres ont des sens différents (sim faible) → polysémie.

    Métriques complémentaires :
    - mean_sim      : similarité moyenne (faible = sens dispersés)
    - min_sim       : similarité minimale (très faible = au moins 2 sens distincts)
    - max_sim       : similarité maximale
    - spread        : max - min (amplitude)
    - n_occurrences : nombre de phrases utilisées
    """
    sims = similarites_cosinus_paires(embeddings)
    return {
        "score_polysemie": float(np.std(sims)),       # critère principal
        "mean_sim":        float(np.mean(sims)),
        "min_sim":         float(np.min(sims)),
        "max_sim":         float(np.max(sims)),
        "spread":          float(np.max(sims) - np.min(sims)),
        "n_occurrences":   len(embeddings),
        "n_paires":        len(sims),
    }


def calculer_scores(embeddings: dict) -> pd.DataFrame:
    """
    Applique score_polysemie à chaque mot et retourne un DataFrame trié.
    """
    lignes = []
    for mot, arr in embeddings.items():
        try:
            metriques = score_polysemie(arr)
            metriques["mot"] = mot
            lignes.append(metriques)
            print(
                f"  {mot:<12} | score={metriques['score_polysemie']:.4f} "
                f"| mean_sim={metriques['mean_sim']:.4f} "
                f"| spread={metriques['spread']:.4f}"
            )
        except ValueError as e:
            print(f"    {mot} ignoré : {e}")

    df = pd.DataFrame(lignes).set_index("mot")
    df.sort_values("score_polysemie", ascending=False, inplace=True)
    return df


def afficher_classement(df: pd.DataFrame):
    """Affiche le classement lisible des mots par score de polysémie."""
    print("\n" + "═" * 65)
    print(f"{'Rang':<5} {'Mot':<14} {'Score (std)':<12} {'Mean sim':<12} {'Spread':<10}")
    print("─" * 65)
    for rang, (mot, row) in enumerate(df.iterrows(), 1):
        barre = "█" * int(row["score_polysemie"] * 200)
        print(
            f"{rang:<5} {mot:<14} {row['score_polysemie']:<12.4f} "
            f"{row['mean_sim']:<12.4f} {row['spread']:<10.4f}  {barre}"
        )
    print("═" * 65)


def sauvegarder_scores(df: pd.DataFrame, dossier: str = "resultats"):
    """Sauvegarde le DataFrame des scores en CSV."""
    Path(dossier).mkdir(exist_ok=True)
    chemin = Path(dossier) / "scores_polysemie.csv"
    df.to_csv(chemin)
    print(f"\n Scores sauvegardés dans « {chemin} »")


def charger_scores(dossier: str = "resultats") -> pd.DataFrame:
    chemin = Path(dossier) / "scores_polysemie.csv"
    return pd.read_csv(chemin, index_col="mot")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from embedding import charger_embeddings

    print(" Chargement des embeddings…")
    embeddings = charger_embeddings("resultats")

    print("\n Calcul des scores de polysémie :\n")
    df_scores = calculer_scores(embeddings)
    afficher_classement(df_scores)
    sauvegarder_scores(df_scores)

    print("\n Scores terminés.")