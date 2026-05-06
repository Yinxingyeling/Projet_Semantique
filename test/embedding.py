"""
Étape 3 — Extraction des embeddings contextuels avec CamemBERT
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import re
# ─────────────────────────────────────────────
# Chargement du corpus depuis le fichier JSON
# Format attendu : { "mot": ["phrase1", "phrase2", ...], ... }
# ─────────────────────────────────────────────
CORPUS_PATH = Path("../clean_corpus_final.json")

def charger_corpus(chemin: Path = CORPUS_PATH) -> dict:
    """Charge le corpus depuis un fichier JSON."""
    with open(chemin, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f" Corpus chargé : {len(corpus)} mots, "
          f"{sum(len(v) for v in corpus.values())} phrases au total")
    for mot, phrases in corpus.items():
        print(f"   {mot:<12} : {len(phrases)} phrases")
    return corpus

CORPUS = charger_corpus()

# ─────────────────────────────────────────────
# Classification linguistique des mots du corpus
# basée sur le nombre de sens attestés (TLFi / Larousse)
# ─────────────────────────────────────────────
MOTS_TRES_POLYSEMIQUES     = ["coup", "temps", "ordre", "feu", "corps",
                               "carte", "lettre", "verre", "rayon", "mousse", "classe"]
MOTS_MOYENNEMENT_POLYSEMIQUES = ["vague", "patron", "feuille", "bouton"]
MOTS_PEU_POLYSEMIQUES      = ["cœur", "maison"]

# Alias binaire pour les analyses de corrélation
MOTS_POLYSEMIQUES = MOTS_TRES_POLYSEMIQUES + MOTS_MOYENNEMENT_POLYSEMIQUES
MOTS_MONOSEMIQUES = MOTS_PEU_POLYSEMIQUES


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def charger_modele(nom_modele: str = "camembert-base"):
    """Charge le tokenizer et le modèle CamemBERT."""
    print(f" Chargement du modèle {nom_modele}…")

    tokenizer = AutoTokenizer.from_pretrained(
        nom_modele,
        use_fast=True
    )

    if not tokenizer.is_fast:
        raise ValueError("Il faut utiliser un fast tokenizer pour return_offsets_mapping=True.")

    model = AutoModel.from_pretrained(
        nom_modele,
        output_hidden_states=True
    )

    model.eval()
    device = get_device()
    model.to(device)

    print(f" Modèle chargé sur {device}")
    return tokenizer, model, device

def trouver_span_mot(phrase: str, mot: str):
    """
    Trouve la première occurrence valide du mot cible dans la phrase.
    Retourne le span caractère (start, end).

    On évite les faux positifs comme :
      - coup dans beaucoup
      - temps dans longtemps
      - verre dans renverre

    Mais on accepte :
      - feuille,
      - (feuille)
      - feuille.
      - bouton-d'or  -> le span retourné correspond seulement à bouton
    """
    pattern = re.compile(
        rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(mot)}(?![A-Za-zÀ-ÖØ-öø-ÿ])",
        re.IGNORECASE
    )

    match = pattern.search(phrase)

    if match is None:
        return None

    return match.start(), match.end()


def trouver_positions_token_offset(tokenizer, phrase: str, mot: str, max_length: int = 512):
    """
    Retourne les indices des sous-tokens correspondant au mot cible,
    en utilisant offset_mapping.

    Contrairement à la méthode qui reconstruit les tokens,
    cette version ne retourne que les sous-tokens qui chevauchent
    exactement la zone caractère du mot cible.

    Exemple :
      phrase = "Il cueille un bouton-d'or."
      mot = "bouton"

    Elle retourne seulement le token de "bouton",
    pas les tokens de "-d'or".
    """
    span = trouver_span_mot(phrase, mot)

    if span is None:
        return []

    start, end = span

    encoded = tokenizer(
        phrase,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length
    )

    offsets = encoded["offset_mapping"][0].tolist()

    positions = []

    for i, (tok_start, tok_end) in enumerate(offsets):
        # tokens spéciaux : <s>, </s>, padding éventuel
        if tok_start == tok_end:
            continue

        # token et mot cible ont une intersection
        if tok_start < end and tok_end > start:
            positions.append(i)

    return positions


def extraire_embedding(
    tokenizer,
    model,
    device,
    phrase: str,
    mot: str,
    couches: list = [-1, -2, -3, -4],
):
    """
    Extrait l'embedding contextuel du mot dans la phrase.

    Stratégie :
      1. trouver le span caractère du mot cible dans la phrase ;
      2. utiliser offset_mapping pour retrouver les sous-tokens correspondants ;
      3. moyenner les 4 dernières couches cachées ;
      4. moyenner les sous-tokens du mot cible.

    Retourne None si le mot n'est pas trouvé.
    """
    encoding = tokenizer(
        phrase,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512
    )

    offsets = encoding.pop("offset_mapping")[0].tolist()

    span = trouver_span_mot(phrase, mot)

    if span is None:
        return None

    start, end = span

    positions = []

    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue

        if tok_start < end and tok_end > start:
            positions.append(i)

    if not positions:
        return None

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        sorties = model(**encoding)

    hidden_states = sorties.hidden_states

    vecteurs_couches = torch.stack(
        [hidden_states[c] for c in couches],
        dim=0
    )

    moyenne_couches = vecteurs_couches.mean(dim=0).squeeze(0)

    embedding_mot = moyenne_couches[positions].mean(dim=0)

    return embedding_mot.cpu().numpy()


def extraire_tous_embeddings(tokenizer, model, device, corpus: dict):
    """
    Parcourt le corpus et extrait un embedding par occurrence.
    Retourne un dict {mot: np.array de shape (n_occurrences, 768)}.
    """
    embeddings = {}
    for mot, phrases in corpus.items():
        print(f"\n Traitement de « {mot} » ({len(phrases)} phrases)…")
        vecteurs = []
        for i, phrase in enumerate(phrases):
            emb = extraire_embedding(tokenizer, model, device, phrase, mot)
            if emb is not None:
                vecteurs.append(emb)
            else:
                print(f"     Mot non trouvé dans : {phrase[:60]}…")
        embeddings[mot] = np.array(vecteurs)
        print(f"   → {len(vecteurs)} embeddings extraits (dim={embeddings[mot].shape})")
    return embeddings


def sauvegarder_embeddings(embeddings: dict, dossier: str = "resultats"):
    """Sauvegarde les embeddings dans des fichiers .npy."""
    Path(dossier).mkdir(exist_ok=True)
    for mot, arr in embeddings.items():
        chemin = Path(dossier) / f"embeddings_{mot}.npy"
        np.save(chemin, arr)
    # Sauvegarder aussi les métadonnées
    meta = {mot: arr.shape[0] for mot, arr in embeddings.items()}
    with open(Path(dossier) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n Embeddings sauvegardés dans « {dossier}/ »")


def charger_embeddings(dossier: str = "resultats"):
    """Recharge les embeddings depuis le disque."""
    dossier = Path(dossier)
    embeddings = {}
    for fichier in sorted(dossier.glob("embeddings_*.npy")):
        mot = fichier.stem.replace("embeddings_", "")
        embeddings[mot] = np.load(fichier)
    print(f" {len(embeddings)} mots chargés depuis « {dossier}/ »")
    return embeddings


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer, model, device = charger_modele("camembert-base")
    embeddings = extraire_tous_embeddings(tokenizer, model, device, CORPUS)
    sauvegarder_embeddings(embeddings)
    print("\n Étape 3 terminée.")
