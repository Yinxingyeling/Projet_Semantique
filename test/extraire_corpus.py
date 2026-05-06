from datasets import load_dataset
import re
import random
import json
from tqdm import tqdm


def clean_sentence_light(sentence):

    sentence = re.sub(r"\s+", " ", sentence).strip()

    sentence = re.sub(
        r"^(Description|Synopsis|Biographie|Filmographie|Historique|Histoire|Carrière|Œuvre|Œuvres|Travaux|Distinctions|Récompenses|Liens externes|Annexes)\s+",
        "",
        sentence
    )

    sentence = re.sub(r"^[Ll]e\s*,\s*", "", sentence)
    sentence = re.sub(r"^[Ee]n\s*,\s*", "", sentence)

    sentence = re.sub(r"\b[Ll]e\s*,\s*", "", sentence)
    sentence = re.sub(r"\b[Ee]n\s*,\s*", "", sentence)

    sentence = re.sub(r"\bde\s+de\b", "de", sentence)
    sentence = re.sub(r"\bà\s+à\b", "à", sentence)

    sentence = re.sub(r"\s+", " ", sentence).strip()

    sentence = re.sub(r"\s+([,.;:!?])", r"\1", sentence)

    return sentence


def clean_sentence(sn, seen_sentences, bad_patterns):

    sn_low = sn.lower()

    if any(p in sn_low for p in bad_patterns):
        return False

    if len(sn) < 50 or len(sn) > 250:
        return False

    if len(sn.split()) < 8:
        return False

    if sn.isupper() or sn.strip().endswith(":"):
        return False

    if sn in seen_sentences:
        return False
    seen_sentences.add(sn)

    return True


def fast_collect():

    words = [
        "patron", "bouton", "feuille",
        "coup", "cœur", "maison", "vague",
        "rayon", "verre", "lettre", "mousse",
        "corps", "carte", "ordre", "temps",
        "classe", "feu"
    ]

    corpus = {w: set() for w in words}

    seen_sentences = set()

    limit = 50
    per_doc_limit = 2

    bad_patterns = [
        "statistiques", "saison", "classement",
        "références", "épisodes", "voir",
        "pour les significations", "modifier",
        "catégorie", "hockey sur glace",
        "liste", "équipe", "joueur"
    ]

    print("Loading Wikipedia dataset...")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.fr",
        split="train",
        streaming=True
    ).shuffle(seed=42, buffer_size=1000)

    pbar = tqdm(total=len(words) * limit)

    for entry in dataset:

        text = entry["text"].replace("\n", " ")
        sentences = re.split(r'(?<=[.!?])\s+', text)

        sentences = [s.strip() for s in sentences if s.strip()]
        sampled = random.sample(sentences, min(len(sentences), 15))

        doc_count = {w: 0 for w in words}

        for sn in sampled:

            if not clean_sentence(sn, seen_sentences, bad_patterns):
                continue

            sn = clean_sentence_light(sn)
            sn_low = sn.lower()

            for w in words:

                if len(corpus[w]) >= limit:
                    continue

                if doc_count[w] >= per_doc_limit:
                    continue

                if re.search(rf'(?<!\w){re.escape(w)}(?!\w)', sn_low):
                    corpus[w].add(sn)
                    doc_count[w] += 1
                    pbar.update(1)

        if all(len(corpus[w]) >= limit for w in words):
            print("\nAll words collected successfully!")
            break

    pbar.close()

    corpus = {w: list(v) for w, v in corpus.items()}

    for w, sents in corpus.items():
        if len(sents) < limit:
            print(f"Warning: '{w}' only has {len(sents)} sentences")

    with open("clean_corpus_final.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("Done! Clean corpus ready for BERT.")


if __name__ == "__main__":
    fast_collect()
