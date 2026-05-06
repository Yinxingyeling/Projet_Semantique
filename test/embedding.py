"""
Étape 3 — Extraction des embeddings contextuels avec CamemBERT
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

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
    """Charge le tokenizer et le modèle BERT."""
    print(f" Chargement du modèle {nom_modele}…")
    tokenizer = AutoTokenizer.from_pretrained(nom_modele)
    model = AutoModel.from_pretrained(nom_modele, output_hidden_states=True)
    model.eval()
    device = get_device()
    model.to(device)
    print(f" Modèle chargé sur {device}")
    return tokenizer, model, device


def trouver_positions_token(tokenizer, phrase: str, mot: str):
    """
    Retourne les indices (dans la séquence tokenisée, +1 pour [CLS])
    des sous-tokens correspondant à une occurrence du mot cible.

    Utilise re.search avec \\b (frontière de mot) sur chaque token reconstruit.
    Gère naturellement :
      - la casse       : "Bouton" → "bouton"
      - la ponctuation : "feuille," "(feuille)" "feuille."
      - les composés   : "bouton-d'or", "branche-et-feuille"
      - les faux positifs : "verre" ne matche pas "renversé"

    Retourne la PREMIÈRE occurrence valide trouvée.
    """
    import re
    tokens = tokenizer.tokenize(phrase)
    mot_lower = mot.lower()
    # Pattern : mot cible entre frontières de mots (non-alphanumériques)
    pattern = re.compile(r'\b' + re.escape(mot_lower) + r'\b', re.IGNORECASE)

    i = 0
    while i < len(tokens):
        # Reconstituer la forme complète depuis les sous-tokens
        subtokens = [tokens[i].lstrip("▁")]
        j = i + 1
        while j < len(tokens) and not tokens[j].startswith("▁"):
            subtokens.append(tokens[j])
            j += 1
        forme = "".join(subtokens)

        if pattern.search(forme):
            return list(range(i + 1, j + 1))  # +1 pour [CLS]
        i = j

    return []


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
    Stratégie : moyenne des 4 dernières couches cachées,
    puis moyenne sur les sous-tokens du mot cible.
    Retourne None si le mot n'est pas trouvé sous aucune forme.
    """
    encoding = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=512)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    positions = trouver_positions_token(tokenizer, phrase, mot)

    if not positions:
        return None

    with torch.no_grad():
        sorties = model(**encoding)

    # hidden_states : tuple de (n_couches+1) tenseurs de shape (1, seq_len, 768)
    hidden_states = sorties.hidden_states  # inclut la couche d'embedding initiale

    # Sélectionner les couches et calculer la moyenne
    vecteurs_couches = torch.stack(
        [hidden_states[c] for c in couches], dim=0
    )  # (n_couches, 1, seq_len, 768)
    moyenne_couches = vecteurs_couches.mean(dim=0).squeeze(0)  # (seq_len, 768)

    # Moyenne sur les sous-tokens du mot cible
    embedding_mot = moyenne_couches[positions].mean(dim=0)  # (768,)
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