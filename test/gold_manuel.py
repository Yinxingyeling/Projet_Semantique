"""
Étape 6 — Mesure de référence linguistique.
Gold standard basé sur les dictionnaires français (TLFi, Larousse) :
nombre d'acceptions distinctes recensées pour chaque mot du corpus.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
# Scores basés sur les dictionnaires français
# Source : TLFi (Trésor de la Langue Française informatisé) et Larousse
# Chaque entrée = nombre de sens / acceptions distincts recensés
# ─────────────────────────────────────────────
SCORES_MANUELS = {
    "patron":  5,   # employeur, saint patron, modèle de couture, skipper, protecteur
    "bouton":  5,   # vêtement, acné, interrupteur, bourgeon, sonnette
    "feuille": 5,   # végétal, papier, journal, feuille d'imposition, tôle
    "coup":    12,  # frappe, bruit, fois, tentative, influence, coup d'État...
    "cœur":    6,   # organe, sentiment, courage, carte, centre, estomac
    "maison":  4,   # bâtiment, famille, entreprise, fait maison
    "vague":   7,   # onde, imprécis, mouvement de foule, nouvelle vague, terrain vague...
    "rayon":   6,   # lumière, géométrie, linéaire de supermarché, roue, département, ruche
    "verre":   5,   # matière, récipient, contenu, lunettes, vitre
    "lettre":  6,   # courrier, caractère typographique, littérature, lettre de change, au pied de la lettre
    "mousse":  5,   # végétal, écume, matériau, marin débutant, entremets
    "corps":   8,   # anatomie, cadavre, groupe armé, substance chimique, corps enseignant, typographie...
    "carte":   6,   # géographique, identité, menu, bancaire, jeu de cartes, carte blanche
    "ordre":   7,   # commandement, organisation, religieux, chevalerie, classement, ordre du jour...
    "temps":   6,   # durée, météo, époque, grammaire, mesure musicale, mi-temps
    "classe":  6,   # salle, niveau scolaire, catégorie sociale, élégance, taxonomie, année militaire
    "feu":     7,   # combustion, lumière, signalisation, défunt, tir d'arme, ardeur, feu de camp
}


def construire_gold_standard(mots: list, dossier: str = "resultats") -> pd.DataFrame:
    """
    Construit le gold standard à partir des scores manuels TLFi/Larousse.
    Normalise les scores en [0, 1] via min-max pour la comparaison avec le score BERT.
    """
    Path(dossier).mkdir(exist_ok=True)
    lignes = []

    print("\n Construction du gold standard (TLFi / Larousse) :\n")
    for mot in mots:
        n_sens = SCORES_MANUELS.get(mot)
        if n_sens is None:
            print(f"    « {mot} » absent de SCORES_MANUELS — à compléter")
        lignes.append({"mot": mot, "n_sens_manuel": n_sens})

    df = pd.DataFrame(lignes).set_index("mot")

    # Normalisation min-max → score_gold dans [0, 1]
    g = df["n_sens_manuel"].dropna()
    if g.max() != g.min():
        df["score_gold"] = (df["n_sens_manuel"] - g.min()) / (g.max() - g.min())
    else:
        df["score_gold"] = 0.5

    chemin = Path(dossier) / "gold_standard.csv"
    df.to_csv(chemin)
    print(f" Gold standard sauvegardé : {chemin}")
    return df


def afficher_gold_standard(df: pd.DataFrame):
    """Affiche le classement des mots par nombre de sens."""
    print("\n" + "═" * 55)
    print(f"{'Rang':<6} {'Mot':<14} {'Sens (TLFi)':<14} {'Gold norm.'}")
    print("─" * 55)
    for rang, (mot, row) in enumerate(df.sort_values("score_gold", ascending=False).iterrows(), 1):
        barre = "█" * int((row["score_gold"] or 0) * 25)
        print(f"{rang:<6} {mot:<14} {str(row['n_sens_manuel']):<14} {row['score_gold']:.3f}  {barre}")
    print("═" * 55)


def charger_gold_standard(dossier: str = "resultats") -> pd.DataFrame:
    chemin = Path(dossier) / "gold_standard.csv"
    return pd.read_csv(chemin, index_col="mot")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from embedding import CORPUS

    mots = list(CORPUS.keys())
    df_gold = construire_gold_standard(mots)
    afficher_gold_standard(df_gold)

    print("\n Étape 6 terminée.")