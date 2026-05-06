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
# Source : TLFi (Trésor de la Langue Française informatisé), Larousse et Robert
# Chaque entrée = nombre de sens / acceptions distincts recensés
# ─────────────────────────────────────────────
# Gold standard manuel — macro-sens
# Sens regroupés à partir du TLFi / Larousse/ Robert
# ─────────────────────────────────────────────

SCORES_MANUELS = {
    "coup":    12,  # choc/frappe; blessure; arme à feu; épreuve; mouvement; action avec objet; signal/bruit; événement soudain; émotion soudaine; jeu/sport; action stratégique; quantité/fois/locutions

    "ordre":   9,  # organisation/disposition; ordre mathématique; ordre social/public; norme/conformité; catégorie/classement; valeur/qualité; architecture/taxonomie; ordre religieux/honorifique; commandement/mot d'ordre

    "temps":   8,  # durée; temps disponible/mesurable; période/moment; époque historique; météo; temps grammatical; mesure musicale/sportive; moment opportun/locutions

    "corps":   8,  # organisme; cadavre; tronc/partie du corps; personne/individu; partie principale d'un objet/texte; substance/objet physique; groupe institutionnel/militaire; typographie/mathématiques

    "feu":     8,  # combustion/flamme/chaleur; incendie/destruction; cuisson/foyer domestique; arme/tir/détonation; lumière/signal/éclat; ardeur/passion; feu d'artifice/camp; défunt

    "carte":   8,  # support papier/carton; carte à jouer; menu; document personnel/administratif/bancaire; carte postale/visite; carte informatique; carte géographique; schéma/représentation

    "rayon":   7,  # lumière; radiation/rayons X; géométrie; roue/disposition radiale; zone/distance d'action; ruche; étagère/domaine commercial

    "vague":   7,  # onde d'eau; mouvement collectif/afflux; phénomène météorologique; forme ondulée; imprécis/flou; ample/non ajusté; emploi nominal/nerf vague

    "classe":  7,  # catégorie; classe sociale; classe biologique/taxonomique; rang/niveau; distinction/allure; groupe d'élèves/cours/salle; contingent militaire

    "cœur":    6,  # organe; centre/partie centrale; essentiel/moment fort; affectivité/amour/bonté; moral/courage/intériorité; symbole/cartes/locution par cœur

    "lettre":  6,  # caractère alphabétique; typographie; forme littérale d'un texte; courrier/correspondance; document officiel/administratif; lettres/culture humaniste

    "patron":  5,  # saint/protecteur; chef/employeur; responsable institutionnel/professionnel; modèle/gabarit; patron de recherche/travaux intellectuels

    "feuille": 5,  # organe végétal; papier/page; document/journal; plaque mince de matière; oreille/sens figuré ou spécialisé

    "bouton":  5,  # bourgeon floral; lésion cutanée; pièce de vêtement; commande mécanique/électrique; bouton informatique

    "verre":   5,  # matière; objet/pièce en verre; récipient; contenu/boisson; verre optique/lunettes/vitre

    "mousse":  5,  # plante; écume/bulles; boisson/bière; préparation alimentaire; matière spongieuse/synthétique

    "maison":  5,  # bâtiment d'habitation; foyer/domicile; service domestique/vie domestique; entreprise/institution; famille/lignée/adjectif maison
}


def construire_gold_standard(mots: list, dossier: str = "resultats") -> pd.DataFrame:
    """
    Construit le gold standard à partir des scores manuels TLFi/Larousse/Robert.
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
