"""
Étape 6 — Mesure de référence linguistique.
Stratégie en cascade :
  1. WOLF (WordNet du français) via fichier XML si disponible
  2. WordNet anglais via NLTK + traduction manuelle
  3. Score manuel si tout échoue (fallback)
"""

import json
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
# Traduction française → anglaise pour WordNet EN
# ─────────────────────────────────────────────
TRADUCTIONS_EN = {
    "patron":  "patron",   # patron, boss, template
    "bouton":  "button",   # button, pimple, bud
    "feuille": "leaf",     # leaf, sheet, page
    "coup":    "blow",     # blow, hit, stroke, move, shot...
    "cœur":    "heart",    # heart (anatomique + figuré)
    "maison":  "house",    # house, home
    "vague":   "wave",     # wave, vague, surge
    "rayon":   "ray",      # ray, shelf, radius, department
    "verre":   "glass",    # glass (matière, récipient, lunettes)
    "lettre":  "letter",   # letter (courrier + alphabet + littérature)
    "mousse":  "foam",     # foam, moss, cabin boy
    "corps":   "body",     # body, corps, substance
    "carte":   "card",     # card, map, menu, chart
    "ordre":   "order",    # order, command, organization, religious order
    "temps":   "time",     # time, weather, tense (grammaire)
    "classe":  "class",    # class, classroom, elegance, grade
    "feu":     "fire",     # fire, light, late (défunt), feu tricolore
}

# Scores manuels basés sur les dictionnaires FR (TLFi, Larousse)
# Nombre de sens / acceptions distincts recensés
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


def score_wordnet_anglais(mot_fr: str, traductions: dict) -> dict:
    """
    Compte le nombre de synsets WordNet pour la traduction anglaise du mot.
    Retourne aussi le nombre de lemmes (formes) et de définitions.
    """
    try:
        import nltk
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")  # vérifier si les données sont là
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet as wn

        mot_en = traductions.get(mot_fr, mot_fr)
        synsets = wn.synsets(mot_en)

        if not synsets:
            return {"source": "wordnet_en", "n_synsets": None, "definitions": []}

        definitions = [s.definition() for s in synsets]
        return {
            "source":      "wordnet_en",
            "mot_en":      mot_en,
            "n_synsets":   len(synsets),
            "n_lemmes":    sum(len(s.lemmas()) for s in synsets),
            "definitions": definitions[:5],  # les 5 premiers
        }
    except Exception as e:
        return {"source": "wordnet_en", "erreur": str(e), "n_synsets": None}


def score_manuel(mot: str, scores: dict) -> dict:
    """Retourne le score manuel basé sur les dictionnaires français."""
    n = scores.get(mot)
    if n is None:
        return {"source": "manuel", "n_sens": None}
    return {"source": "manuel", "n_sens": n}


def construire_gold_standard(mots: list, dossier: str = "resultats") -> pd.DataFrame:
    """
    Pour chaque mot, récupère :
    - n_synsets_en  : nombre de synsets WordNet anglais
    - n_sens_manuel : nombre de sens selon les dictionnaires français
    - score_gold    : combinaison normalisée (moyenne des deux)
    """
    Path(dossier).mkdir(exist_ok=True)
    lignes = []

    print("\n Construction du gold standard linguistique :\n")
    for mot in mots:
        row = {"mot": mot}

        # WordNet anglais
        wn_res = score_wordnet_anglais(mot, TRADUCTIONS_EN)
        row["n_synsets_en"]  = wn_res.get("n_synsets")
        row["mot_en"]        = wn_res.get("mot_en", TRADUCTIONS_EN.get(mot, "?"))
        row["definitions_wn"] = " | ".join(wn_res.get("definitions", [])[:3])

        # Score manuel
        man_res = score_manuel(mot, SCORES_MANUELS)
        row["n_sens_manuel"] = man_res.get("n_sens")

        # Score gold combiné (normalisation min-max)
        valeurs = [v for v in [row["n_synsets_en"], row["n_sens_manuel"]] if v is not None]
        row["score_gold_brut"] = np.mean(valeurs) if valeurs else None

        print(
            f"  {mot:<12} | synsets_EN={str(row['n_synsets_en']):<6} "
            f"| sens_FR={str(row['n_sens_manuel']):<4} "
            f"| gold_brut={str(round(row['score_gold_brut'], 2)) if row['score_gold_brut'] else 'N/A'}"
        )
        lignes.append(row)

    df = pd.DataFrame(lignes).set_index("mot")

    # Normalisation min-max du score gold
    g = df["score_gold_brut"].dropna()
    if g.max() != g.min():
        df["score_gold"] = (df["score_gold_brut"] - g.min()) / (g.max() - g.min())
    else:
        df["score_gold"] = 0.5

    # Sauvegarder
    chemin = Path(dossier) / "gold_standard.csv"
    df.to_csv(chemin)
    print(f"\n Gold standard sauvegardé : {chemin}")
    return df


def afficher_gold_standard(df: pd.DataFrame):
    """Affiche le tableau du gold standard de manière lisible."""
    print("\n" + "═" * 75)
    print(f"{'Mot':<14} {'Synsets EN':<13} {'Sens FR':<12} {'Gold brut':<12} {'Gold norm.':<10}")
    print("─" * 75)
    df_trie = df.sort_values("score_gold", ascending=False)
    for mot, row in df_trie.iterrows():
        barre = "█" * int((row["score_gold"] or 0) * 20)
        print(
            f"{mot:<14} {str(row['n_synsets_en']):<13} {str(row['n_sens_manuel']):<12} "
            f"{round(row['score_gold_brut'], 2) if row['score_gold_brut'] else 'N/A':<12} "
            f"{round(row['score_gold'], 3) if row['score_gold'] else 'N/A':<10}  {barre}"
        )
    print("═" * 75)


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

    print("\n Gold terminé.")