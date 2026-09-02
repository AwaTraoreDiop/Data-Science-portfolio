Analyse Exploratoire des Données (EDA) et Traitement
Source de Données : Les données ont été collectées via l'API Judilibre (mise à jour avec pyjudilibre pour plus de robustesse) en ciblant les décisions pénales (criminelles et correctionnelles) disponibles depuis fin 2025.
Corpus : Un corpus de décisions a été constitué en recherchant des mots-clés spécifiques pour chaque catégorie d'infraction. La distribution des décisions par catégorie montre une certaine hétérogénéité, avec certaines catégories plus représentées que d'autres.
Longueurs des Décisions : L'analyse des longueurs de texte (nombre de mots) indique que la majorité des décisions ont une longueur modérée, avec une médiane d'environ 73 mots.
Juridiction : Les décisions collectées proviennent principalement de la Cour de cassation.
Catégorisation : Les 107 catégories d'infractions brutes de l'ONDRP (Office National de la Délinquance et des Réponses Pénales) ont été regroupées en 6 catégories thématiques plus larges et équilibrées (ex: "Atteintes aux personnes", "Atteintes aux biens"). Ces catégories ont été mappées à des identifiants numériques pour l'entraînement des modèles.
Préparation du Modèle : Les données ont été divisées en ensembles d'entraînement et de test avec une stratification pour assurer une distribution similaire des catégories dans les deux ensembles.
Comparaison des Modèles (TF-IDF+LogReg vs. CamemBERT)
Deux approches de classification ont été utilisées et comparées :

Baseline (TF-IDF + Régression Logistique) :
Ce modèle utilise une vectorisation des textes basée sur la fréquence des termes (TF-IDF) combinée à une régression logistique.
Il a obtenu un F1-score pondéré de 0.79.
Modèle Avancé (CamemBERT fine-tuné) :
Le modèle CamemBERT, un modèle de langage pré-entraîné pour le français, a été fine-tuné sur le corpus de décisions.
Il a atteint un F1-score pondéré de 0.73.
Conclusion sur les modèles : De manière surprenante, le modèle baseline (TF-IDF + Régression Logistique) a surpassé le modèle CamemBERT fine-tuné de +5.6 points de F1. Cela peut s'expliquer par la taille relativement petite du corpus, où des méthodes de machine learning plus simples peuvent parfois performer aussi bien, voire mieux, que des modèles profonds qui nécessitent de très grandes quantités de données pour montrer leur plein potentiel. D'autres facteurs comme les hyperparamètres de fine-tuning ou la spécificité du langage juridique pourraient aussi jouer un rôle.

Entités Nommées (NER)
L'extraction d'entités nommées avec spaCy a révélé :

Types d'entités les plus fréquents : MISC (divers), ORG (organisations), LOC (lieux) et PER (personnes). Cela est cohérent avec le type de texte (décisions juridiques).
Entités les plus mentionnées : On retrouve fréquemment des termes comme "Code pénal", "Cour de Cassation", "Union européenne", "TVA", "Code des douanes", ainsi que diverses institutions et codes législatifs. Ceci confirme la richesse terminologique juridique des textes analysés.
