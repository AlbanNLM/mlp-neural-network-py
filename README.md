# mlp-neural-network-py

Ce projet pédagogique a pour but d'implémenter un réseau de neurones multicouches de zéro ("from scratch") en utilisant la Programmation Orientée Objet (POO) de Python, en commentant un maximum chaque étape du code.

Il s'adresse principalement aux personnes ayant déjà des bases théoriques en réseaux de neurones et qui souhaiteraient un exemple concret, détaillé et récapitulatif de leur fonctionnement interne.

## Concepts abordés

Ce fichier vous permet de plonger dans l'implémentation des concepts fondamentaux du Deep Learning :

- Architecture dynamique : Création d'un réseau avec un nombre de couches et de neurones personnalisable.

- Fonctions d'activation : Implémentation mathématique de ReLU (pour les couches cachées) et Sigmoïde (pour la sortie binaire).

- Le cycle d'apprentissage :

    1. Forward Propagation (Propagation avant)

    2. Loss Function (Calcul de l'erreur via l'Entropie Croisée Binaire)

    3. Backward Propagation (Rétropropagation du gradient avec la règle de la chaîne)

    4. Gradient Descent (Mise à jour des poids et biais)

- Optimisation : Utilisation de NumPy pour vectoriser les calculs matriciels et éviter les boucles chronophages.

## Utilisation

Le script est interactif et se suffit à lui-même. À la fin du fichier, un cas d'usage concret est implémenté : la résolution du problème logique XOR, prouvant l'efficacité des couches cachées et de la non-linéarité.

1. Clonez le dépôt

2. Installez les dépendances requises :
```
pip install numpy
```

3. Exécutez le script pour lancer l'entraînement et voir les prédictions :
```
python mlp_from_scratch.py
```
