# Classification automatique de décisions judiciaires françaises par type d'infraction

## Description
Pipeline NLP complet combinant deux sources de données publiques françaises :
- **Labels** : 107 catégories d'infractions ONDRP / SSMSI (data.gouv.fr)
- **Textes** : Décisions de justice via l'API Judilibre (Cour de cassation)

## Objectif
Classifier automatiquement une décision judiciaire par type d'infraction
sans intervention humaine, en utilisant CamemBERT fine-tuné.

## Stack technique
- Python, HuggingFace Transformers, CamemBERT
- spaCy (NER), scikit-learn (baseline TF-IDF)
- API Judilibre, data.gouv.fr

## Résultats
| Modèle | F1-score |
|--------|----------|
| Baseline TF-IDF + LogReg | X.XX |
| CamemBERT fine-tuné | X.XX |

## Application directe
Détection de fraudes (DGDDI), analyse de signalements (sécurité publique)

## Structure
├── notebook.ipynb
├── data/
│   └── corpus_judilibre.csv
├── results/
│   ├── comparaison_modeles.png
│   └── ner_analyse.png
└── README.md
