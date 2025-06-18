"""
This script performs the preprocessing of the texts using spaCy for Romanian language processing.

First, download the spacy model:
```bash
python -m spacy download ro_core_news_lg
```

Then, run from the root of the project to generate the anonymized:

```bash
python ./satire/dataset/preprocessing.py \
    --proc_type anonymize \
    -i ./data/csv/selerosa_raw.csv \
    -o ./data/csv/selerosa_anonymized.csv
```

or processed dataset:

```bash
python ./satire/dataset/preprocessing.py \
    --proc_type full \
    -i ./data/csv/selerosa_raw.csv \
    -o ./data/csv/selerosa_proc.csv
```
"""

import os
import re
from multiprocessing import Pool

import spacy
from tqdm import tqdm

# Get the directory of this script to find the stopwords file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


nlp = spacy.load("ro_core_news_lg")


def load_stopwords() -> set[str]:
    """
    Load custom Romanian stopwords from the data file and normalize characters.

    Returns:
        set[str]: A set of normalized stopwords.
    """
    with open(os.path.join(SCRIPT_DIR, "../tools/data/stopwords.txt"), "r") as f:
        stop_words = f.read().splitlines()
    stop_words = list(
        map(
            lambda x: x.replace("ţ", "ț").replace("ş", "ș").strip(),
            stop_words,
        )
    )
    return set(stop_words)


def anonymize_text(
    text: str,
    labels_to_anonymize: set[str] | None = None,
) -> str:
    """
    Anonymize the input text by replacing named entities such as person names, locations,
    organizations, etc., with placeholder tokens.

    Parameters:
        text (str): The original text.

    Returns:
        str: The text with sensitive entities replaced by placeholders.
    """
    if labels_to_anonymize is None:
        labels_to_anonymize = {"PERSON", "NORP", "GPE", "ORG", "LOC", "FAC"}

    doc = nlp(text)

    # Define entity labels to anonymize.
    anonymized_text = text

    # Replace entities in reverse order to maintain correct character indices.
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in labels_to_anonymize:
            placeholder = f"<{ent.label_}>"
            anonymized_text = (
                anonymized_text[: ent.start_char]
                + placeholder
                + anonymized_text[ent.end_char :]
            )

    anonymized_text = re.sub(r"TNR", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"Google", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"BitDefender", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"Bitdefender", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"Facebook", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"RDS", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"Samsung", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"Apple", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"Youtube", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"YouTube", "<ORG>", anonymized_text)
    anonymized_text = re.sub(r"Timesnewroman.ro", "@URL", anonymized_text)
    anonymized_text = re.sub(r"www.cautatoriidepovesti.ro", "@URL", anonymized_text)

    return anonymized_text


def process_text(text: str, labels_to_anonymize: set[str] | None = None) -> str:
    """Process the input text by cleaning, normalizing, and tokenizing it.

    Args:
        text (str): The original text to process.
        labels_to_anonymize (set[str] | None): The set of entity labels to anonymize. This is just to restore them after preprocessing.

    Returns:
        str: The processed text.
    """
    if labels_to_anonymize is None:
        labels_to_anonymize = {"PERSON", "NORP", "GPE", "ORG", "LOC", "FAC"}

    # Step 2: Clean the text (remove emails, extra whitespace, and unwanted punctuation)
    text = re.sub(r"\S*@\S*\s?", "", text)  # Remove email addresses
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace/newlines
    text = text.replace("ţ", "ț").replace("ş", "ș")

    # Step 3: Lowercase the text
    text = text.lower()

    # Step 4: Process with spaCy for tokenization and lemmatization
    doc = nlp(text)

    # Step 5: Remove stopwords and tokens with fewer than 3 characters
    stop_words = load_stopwords()

    tokens = []
    for token in doc:
        if len(token.text) <= 3 and token.text not in ("<", ">"):
            continue  # Skip short tokens except for < >
        if token.is_space or token.is_punct:
            continue  # Skip spaces and punctuation
        if token.is_stop or token.lemma_ in stop_words:
            continue  # Skip stopwords

        # Step 6: Lemmatize tokens
        lemma = token.lemma_.lower()
        tokens.append(lemma)

    text = " ".join(tokens)

    # Readd tags
    text = re.sub(r"< >", "", text)
    for tag in labels_to_anonymize:
        text = re.sub(rf"< {tag.lower()} >", f"<{tag}>", text)

    return text


def preprocess(text: str, labels_to_anonymize: set[str] | None = None) -> str:
    """
    Preprocess the input text by:
      - Anonymizing named entities.
      - Removing email addresses and extra whitespace.
      - Lowercasing the text.
      - Processing with spaCy for tokenization and linguistic analysis.
      - Removing stopwords (custom list + spaCy's built-in) and short tokens.
      - Lemmatizing tokens using spaCy's Romanian lemmatizer.

    Parameters:
        text (str): The original text.

    Returns:
        str: The processed text as a string of tokens.
    """
    # Step 1: Anonymize named entities (e.g., person names, locations, institutions)
    if labels_to_anonymize is None:
        labels_to_anonymize = {"PERSON", "NORP", "GPE", "ORG", "LOC", "FAC"}
    text = anonymize_text(text, labels_to_anonymize)
    text = process_text(text, labels_to_anonymize)
    return text


if __name__ == "__main__":
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Text preprocessing script. Run from the root of the project",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "-p",
        "--proc_type",
        choices=["anonymize", "full"],
        default="anonymize",
        help="Processing type",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    print("Before:")
    print("\n\n====\n\n".join(df["sentence"].tolist()[:10]))

    proc = anonymize_text if args.proc_type == "anonymize" else preprocess

    with Pool(processes=8) as p:
        df["sentence"] = list(
            tqdm(
                p.imap(proc, df["sentence"]),
                total=len(df),
                desc="Processing",
            )
        )

    print("After:")
    print("\n\n====\n\n".join(df["sentence"].tolist()[:10]))

    df.to_csv(args.output_file, index=False)
