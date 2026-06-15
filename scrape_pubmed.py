"""
scrape_pubmed.py  —  PubMed NLP Corpus Builder for AMR India
=============================================================
What this script does:
  - Searches PubMed for AMR literature about India
  - Downloads titles + abstracts
  - Extracts mentions of pathogens, antibiotics, resistance keywords
  - Saves a clean NLP corpus CSV for analysis

What this script does NOT do:
  - It does NOT create fake patient records
  - It does NOT touch master_amr_icmr.csv or the clinical pipeline
  - It does NOT feed into train_model.py

Output:  processed/nlp_corpus.csv
Columns: pmid | year | title | abstract | pathogens_found |
         antibiotics_found | resistance_keywords | sentence_count

Run: python scrape_pubmed.py
"""

import os
import re
import pandas as pd
from Bio import Entrez
import xml.etree.ElementTree as ET

# ── Your email (required by NCBI) ─────────────────────────────────────────────
Entrez.email = "smeharpreet@example.com"   # replace with your real email

os.makedirs("processed", exist_ok=True)

# ── Entity lists for rule-based NER ──────────────────────────────────────────
PATHOGENS = [
    "Escherichia coli", "E. coli",
    "Klebsiella pneumoniae", "Klebsiella",
    "Staphylococcus aureus", "S. aureus", "MRSA",
    "Acinetobacter baumannii", "Acinetobacter",
    "Pseudomonas aeruginosa", "Pseudomonas",
    "Enterobacter cloacae", "Enterobacter",
    "Burkholderia cepacia",
    "Streptococcus pneumoniae",
    "Salmonella", "Shigella",
]

ANTIBIOTICS = [
    "carbapenem", "meropenem", "imipenem", "ertapenem",
    "cephalosporin", "cefotaxime", "ceftazidime", "ceftriaxone", "cefepime",
    "fluoroquinolone", "ciprofloxacin", "levofloxacin",
    "amikacin", "gentamicin", "tobramycin",
    "colistin", "polymyxin",
    "piperacillin", "ampicillin",
    "vancomycin", "linezolid", "daptomycin",
    "azithromycin", "tetracycline", "tigecycline",
    "trimethoprim", "sulfamethoxazole",
]

RESISTANCE_KEYWORDS = [
    "multidrug resistant", "MDR",
    "extensively drug resistant", "XDR",
    "pan drug resistant", "PDR",
    "carbapenem resistant", "CRE",
    "ESBL", "extended spectrum beta-lactamase",
    "methicillin resistant", "MRSA",
    "vancomycin resistant", "VRE",
    "antibiotic resistance", "antimicrobial resistance", "AMR",
    "resistance pattern", "susceptibility",
    "MIC", "minimum inhibitory concentration",
]


def extract_entities(text):
    """Return comma-separated lists of matched entities from text."""
    text_lower = text.lower()

    found_pathogens = list({
        p for p in PATHOGENS
        if p.lower() in text_lower
    })

    found_antibiotics = list({
        a for a in ANTIBIOTICS
        if a.lower() in text_lower
    })

    found_keywords = list({
        k for k in RESISTANCE_KEYWORDS
        if k.lower() in text_lower
    })

    return (
        "; ".join(sorted(found_pathogens)),
        "; ".join(sorted(found_antibiotics)),
        "; ".join(sorted(found_keywords)),
    )


def fetch_corpus(max_records=200):
    # ── 1. Search ─────────────────────────────────────────────────────────────
    # Three targeted queries covering clinical isolates, surveillance, and India
    queries = [
        '("antimicrobial resistance" OR "antibiotic resistance") AND "India" AND "clinical isolates"',
        '("MRSA" OR "carbapenem resistant" OR "ESBL") AND "India"',
        '"AMR surveillance" AND "India"',
    ]

    all_ids = set()
    for q in queries:
        handle = Entrez.esearch(db="pubmed", term=q, retmax=max_records // len(queries))
        result = Entrez.read(handle)
        handle.close()
        all_ids.update(result["IdList"])
        print(f"  Query: {q[:60]}...  → {len(result['IdList'])} hits")

    id_list = list(all_ids)
    print(f"\nTotal unique PMIDs: {len(id_list)}")

    if not id_list:
        return pd.DataFrame()

    # ── 2. Fetch XML ──────────────────────────────────────────────────────────
    print("Downloading abstracts...")
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
    xml_data = handle.read()
    handle.close()

    # ── 3. Parse ──────────────────────────────────────────────────────────────
    root = ET.fromstring(xml_data)
    records = []

    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.find(".//PMID").text

            title_node = article.find(".//ArticleTitle")
            title = title_node.text if title_node is not None else ""
            if not title:
                continue

            abstract_nodes = article.findall(".//AbstractText")
            abstract = " ".join(n.text for n in abstract_nodes if n.text)

            # Year — try PubDate/Year, fall back to MedlineDate
            year_node = article.find(".//PubDate/Year")
            if year_node is not None:
                year = year_node.text
            else:
                med_node = article.find(".//PubDate/MedlineDate")
                year = med_node.text[:4] if med_node is not None else "Unknown"

            # Journal
            journal_node = article.find(".//Journal/Title")
            journal = journal_node.text if journal_node is not None else ""

            # Entity extraction on title + abstract combined
            full_text = f"{title} {abstract}"
            pathogens_found, antibiotics_found, resistance_keywords = extract_entities(full_text)

            # Basic stats
            sentence_count = len(re.split(r"[.!?]+", abstract))
            word_count = len(abstract.split())

            records.append({
                "pmid":                pmid,
                "year":                year,
                "journal":             journal,
                "title":               title,
                "abstract":            abstract,
                "word_count":          word_count,
                "sentence_count":      sentence_count,
                "pathogens_found":     pathogens_found,
                "antibiotics_found":   antibiotics_found,
                "resistance_keywords": resistance_keywords,
                # Flags for quick filtering
                "has_india":           "india" in full_text.lower(),
                "has_resistance":      bool(resistance_keywords),
                "has_pathogen":        bool(pathogens_found),
            })

        except Exception:
            continue

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("PubMed AMR India — NLP Corpus Builder")
    print("=" * 50)
    print("NOTE: This script builds a text corpus for NLP analysis.")
    print("      It does NOT create clinical records or touch the model pipeline.")
    print("=" * 50 + "\n")

    df = fetch_corpus(max_records=200)

    if df.empty:
        print("No results returned. Check your internet connection or Entrez.email.")
    else:
        out = "processed/nlp_corpus.csv"
        df.to_csv(out, index=False)

        print(f"\nSaved: {out}")
        print(f"Total articles: {len(df)}")
        print(f"With abstract:  {(df['word_count'] > 0).sum()}")
        print(f"With pathogen mention: {df['has_pathogen'].sum()}")
        print(f"With resistance keyword: {df['has_resistance'].sum()}")
        print(f"\nTop pathogens mentioned:")
        from collections import Counter
        all_pathogens = [p for row in df["pathogens_found"].dropna() for p in row.split("; ") if p]
        for name, count in Counter(all_pathogens).most_common(8):
            print(f"  {name:<35} {count}")
        print(f"\nTop antibiotics mentioned:")
        all_abs = [a for row in df["antibiotics_found"].dropna() for a in row.split("; ") if a]
        for name, count in Counter(all_abs).most_common(8):
            print(f"  {name:<35} {count}")
        print("\nNext step: feed nlp_corpus.csv into a SpaCy or BERT NER pipeline for deeper extraction.")