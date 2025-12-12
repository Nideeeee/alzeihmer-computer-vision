# alzeihmer-computer-vision

Ce projet permet d'entraîner des modèles pour prédire le développement de la maladie d'Alzheimer via des architectures traitant du 2D, 2.5D et 3D. 

# Organisation

Les différentes architectures, entraînements, visualisations et traitement des données sont stockés dans les dossiers 2D, 2.5D et 3D selon leurs dimensions de traitement respectives.
La base de données (post pré-traitement) nous a été transmise par Mr. Clément.
Pour lancer les scripts: ```python fichier.py```

# Différence avec la soutenance

## 2D

La première couche du modèle a changé pour prendre des images noir et blanc plutôt que du RGB (et dupliquer l'image), ainsi la première couche n'est plus gelé lors de l'entraînement.

## 2.5D

Une visualisation t-SNE et UMAP des patients CN/AD en plus des MCI a été ajoutée, mais non présentée car trop d'informations étaient présentes sur le graphique, ce qui le rendait difficilement visible tout en rajoutant peu d'utilité. De plus, les figures ont été retravaillés pour une meilleure visibilité.

## 3D

Une comparaison finetune/non finetune sur les MCI a été implémenté après la soutenance, afin d'obtenir une plus grande transparence pour savoir quid du passage du 2.5D au 3D ou du finetuning est le plus impactant sur la précision du modèle. 