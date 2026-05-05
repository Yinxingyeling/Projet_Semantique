#!/usr/bin/env python3
# python3 script_corpus.py --targets target.txt --output occurrences_wikipedia_api.csv --max-per-word 30 --limit-pages 50 --require-exact
import argparse
import csv
import html
import random
import re
import time
from collections import Counter

import requests
from tqdm import tqdm


API_URL = "https://fr.wikipedia.org/w/api.php"


def request_json(params, max_retries=6):
    """
    Envoie une requête à l'API Wikipédia.
    Si l'API répond 429 Too Many Requests, on attend puis on réessaie.
    """
    headers = {
        "User-Agent": "polysemie-bert-student-project/1.0"
    }

    for attempt in range(max_retries):
        r = requests.get(
            API_URL,
            params=params,
            headers=headers,
            timeout=30,
        )

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")

            if retry_after is not None and retry_after.isdigit():
                pause = int(retry_after)
            else:
                pause = 5 + attempt * 5

            print(f"HTTP 429: trop de requêtes. Pause {pause}s puis nouvel essai...")
            time.sleep(pause)
            continue

        r.raise_for_status()

        # Petite pause pour éviter de surcharger l'API.
        time.sleep(1.0 + random.random() * 0.5)

        return r.json()

    raise RuntimeError("Trop de réponses 429 : l'API Wikipédia bloque encore les requêtes.")


def load_targets(path):
    """
    Lit la liste des mots cibles.
    Un mot par ligne.
    """
    with open(path, encoding="utf-8") as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def split_sentences(text):
    """
    Découpage simple en phrases.
    Suffisant pour une première version du projet.
    """
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def search_pages(word, limit_pages=80):
    """
    Recherche des pages Wikipédia contenant le mot cible.
    Retourne une liste de pageids.
    """
    pageids = []
    offset = 0

    while len(pageids) < limit_pages:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f'"{word}"',
            "srlimit": 20,
            "sroffset": offset,
            "format": "json",
            "utf8": 1,
        }

        data = request_json(params)

        results = data.get("query", {}).get("search", [])
        if not results:
            break

        for item in results:
            pageids.append(str(item["pageid"]))

        if "continue" not in data:
            break

        offset = data["continue"]["sroffset"]

    return pageids[:limit_pages]


def get_extracts(pageids):
    """
    Télécharge le texte brut des pages Wikipédia.
    On récupère les pages par petits paquets pour éviter HTTP 429.
    """
    texts = []

    for i in range(0, len(pageids), 10):
        batch = pageids[i:i + 10]

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "pageids": "|".join(batch),
            "format": "json",
            "utf8": 1,
        }

        data = request_json(params)

        pages = data.get("query", {}).get("pages", {})
        for pageid, page in pages.items():
            title = page.get("title", "")
            extract = page.get("extract", "")

            if extract:
                texts.append((title, extract))

    return texts


def collect_sentences_for_word(word, max_occurrences=50, limit_pages=50):
    """
    Pour un mot cible :
    - cherche des pages Wikipédia
    - récupère leur texte
    - extrait les phrases contenant exactement ce mot
    - garde au maximum max_occurrences phrases
    """
    pattern = re.compile(
        rf"(?<!\w){re.escape(word)}(?!\w)",
        re.IGNORECASE
    )

    pageids = search_pages(word, limit_pages=limit_pages)
    texts = get_extracts(pageids)

    rows = []
    seen_sentences = set()

    for title, text in texts:
        for sent in split_sentences(text):
            sent = html.unescape(sent)
            sent = sent.strip()

            # Filtrage simple des phrases trop courtes ou trop longues.
            if len(sent) < 40 or len(sent) > 300:
                continue

            # Vérifier que la phrase contient bien le mot cible.
            if not pattern.search(sent):
                continue

            # Éviter les doublons.
            key = sent.lower()
            if key in seen_sentences:
                continue

            seen_sentences.add(key)

            rows.append({
                "word": word,
                "sentence": sent,
                "title": title,
                "source": "wikipedia_api",
            })

            if len(rows) >= max_occurrences:
                return rows

    return rows


def filter_words_with_enough_occurrences(rows, min_occurrences):
    """
    Supprime les mots qui ont moins de min_occurrences occurrences.

    Exemple :
    si min_occurrences = 30,
    un mot avec 29 phrases est supprimé entièrement.
    """
    counts = Counter(row["word"] for row in rows)

    kept_words = {
        word
        for word, count in counts.items()
        if count >= min_occurrences
    }

    filtered_rows = [
        row
        for row in rows
        if row["word"] in kept_words
    ]

    return filtered_rows, counts, kept_words


def print_counts(rows, title):
    """
    Affiche le nombre d'occurrences par mot.
    """
    print(f"\n{title}")
    counts = Counter(row["word"] for row in rows)

    for word, count in sorted(counts.items()):
        print(f"{word}: {count} occurrences")

    print(f"Total: {len(rows)} occurrences")


def save_csv(rows, output_path):
    """
    Sauvegarde les occurrences dans un fichier CSV.
    """
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "word", "sentence", "title", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--targets",
        required=True,
        help="Fichier contenant les mots cibles, un mot par ligne"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Fichier CSV de sortie"
    )

    parser.add_argument(
        "--max-per-word",
        type=int,
        default=50,
        help="Nombre maximal d'occurrences à garder par mot"
    )

    parser.add_argument(
        "--limit-pages",
        type=int,
        default=50,
        help="Nombre maximal de pages Wikipédia à consulter par mot"
    )

    # supprimer les mots qui n'ont pas assez d'occurence
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Supprimer les mots qui ont moins de max-per-word occurrences"
    )

    args = parser.parse_args()

    targets = load_targets(args.targets)

    all_rows = []

    for word in tqdm(targets):
        rows = collect_sentences_for_word(
            word,
            max_occurrences=args.max_per_word,
            limit_pages=args.limit_pages,
        )

        print(f"{word}: {len(rows)} occurrences trouvées")

        for i, row in enumerate(rows):
            row["id"] = f"{word}_{i}"

        all_rows.extend(rows)

    print_counts(all_rows, "Occurrences avant filtrage")

    if args.require_exact:
        all_rows, counts_before, kept_words = filter_words_with_enough_occurrences(
            all_rows,
            min_occurrences=args.max_per_word
        )

        removed_words = [
            word
            for word, count in counts_before.items()
            if count < args.max_per_word
        ]

        print("\nMots supprimés car occurrences insuffisantes :")
        if removed_words:
            for word in sorted(removed_words):
                print(f"{word}: {counts_before[word]} occurrences")
        else:
            print("Aucun mot supprimé.")

        print_counts(all_rows, "Occurrences après filtrage")

    save_csv(all_rows, args.output)

    print(f"\nFichier sauvegardé : {args.output}")
    print(f"Nombre total d'occurrences sauvegardées : {len(all_rows)}")


if __name__ == "__main__":
    main()
