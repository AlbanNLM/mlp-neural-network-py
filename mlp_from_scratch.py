# Ce document a pour but de montrer comment fonctionne un réseau de neurones multicouches en utilisant la POO de python.
# Il sert surtout à des fins éducatifs, pour implémenter un réseau de neurones dans un projet réel, on utilisera plutôt des bibliothèques de haut niveau comme TensorFlow ou PyTorch.
# Avant de plonger dans le code, je vous conseille de regarder la vidéo suivante qui explique en surface le fonctionnement d'un réseau de neurones multicouches : https://www.youtube.com/watch?v=gPVVsw2OWdM
# Si vous parlez anglais, vous pouvez aussi regarder celle-ci qui est plus détaillée : https://www.youtube.com/watch?v=aircAruvnKk

# Il faut tout d'abord comprendre comment fonctionne un réseau de neurones multicouches (Perceptron Multicouche) :
# - Un réseau de neurones est composé de plusieurs couches : une couche d'entrée, une ou plusieurs couches cachées, et une couche de sortie.
# - Chaque couche est constituée de neurones qui effectuent des calculs sur les données d'entrée et transmettent les résultats à la couche suivante.
# - Les neurones utilisent la fonction d'activation ReLU pour introduire de la non-linéarité dans le modèle, ce qui permet de résoudre des problèmes complexes.
# - La couche de sortie utilise la fonction d'activation Sigmoid pour produire des probabilités entre 0 et 1.
# - Entre chaque couche, les données sont transformées à l'aide de poids et de biais qui sont ajustés pendant l'entraînement du réseau pour minimiser l'erreur de prédiction.

# Le processus d'entraînement du réseau de neurones comprend les étapes suivantes :
# 1. Propagation avant (Forward propagation) : Les données d'entrée sont transmises à travers les couches du réseau pour produire une prédiction.
# 2. Calcul de la fonction de coût (Cost function) : La différence entre les prédictions du réseau et les étiquettes réelles est mesurée à l'aide d'une fonction de coût, comme l'entropie croisée binaire pour les problèmes de classification binaire.
# 3. Rétropropagation (Backward propagation) : Les gradients de la fonction de coût par rapport aux poids et biais du réseau sont calculés en utilisant la règle de la chaîne (Chain rule) pour propager les erreurs à travers les couches du réseau.
# 4. Mise à jour des paramètres (Weights update) : Les poids et biais du réseau sont mis à jour en utilisant l'algorithme de descente de gradient (Gradient Descent) pour minimiser la fonction de coût.

# Pour l'implémentation on utilisera :
# - ReLU pour les couches cachées, car c'est une fonction d'activation efficace pour les réseaux de neurones profonds.
# - Sigmoid pour la couche de sortie, car nous avons un problème de classification binaire.
# - La fonction de coût utilisée sera l'entropie croisée binaire (Binary Cross Entropy) qui est adaptée pour les problèmes de classification binaire.
# - L'initialisation des poids sera de type "He" pour améliorer les performances du réseau avec la fonction d'activation ReLU.
# - La rétropropagation sera optimisée en utilisant la simplification mathématique de la dérivée de l'erreur pour la combinaison Sigmoid + Binary Cross Entropy, ce qui permet d'éviter des calculs redondants et d'améliorer l'efficacité du processus d'entraînement.
# Vous pouvez vous renseigner sur internet pour mieux comprendre comment fonctionnent ces techniques, elles seront cependant expliquées en surface en commentaires.

# Enfin, pour éviter que notre système ne devienne trop lent, nous allons utiliser la bibliothèque NumPy pour les calculs matriciels, ce qui permettra d'optimiser les performances du réseau de neurones.
# On créera notre réseau de neuronnes à l'aide d'une classe Python, ce qui nous permettra de mieux organiser notre code et de réutiliser facilement notre réseau de neurones dans différents projets.

import numpy as np

class ReseauNeurones:
    """
    Classe représentant un réseau de neurones
    """

    def __init__(self, architecture):
        """
        Initialise le réseau de neurones.
        
        Il prends en paramètre une liste d'entiers représentant le nombre de neurones dans chaque couche du réseau.
        Par exemplee : [2, 5, 5, 1] -> 2 entrées, 2 couches cachées de 5, 1 sortie.
        """
        self.architecture = architecture
        self.parametres = {} # Dictionnaire pour stocker les poids (W) et les biais (b) de chaque couche
        self.L = len(architecture) - 1 # Nombre de couches (sans compter la couche d'entrée)
        
        # Initialisation des poids (W) et des biais (b)
        # On utilise une initialisation de type "He" pour de meilleures performances avec ReLU
        # La formule pour l'initialisation de He est : W = np.random.randn(taille_couche_actuelle, taille_couche_precedente) * np.sqrt(2. / taille_couche_precedente)
        # Elle permets d'éviter les problèmes de vanishing/exploding gradients dans les réseaux profonds en adaptant la variance des poids à la taille de la couche précédente.

        np.random.seed(31) # On initialise une seed (n'importe laquelle, ici 31 parce que Toulouse quoi) pour le random pour la reproductibilité
        for l in range(1, self.L + 1):
            # W (poids) de dimension (taille_couche_actuelle, taille_couche_precedente)
            self.parametres['W' + str(l)] = np.random.randn(architecture[l], architecture[l-1]) * np.sqrt(2. / architecture[l-1])
            # b (biais) de dimension (taille_couche_actuelle, 1)
            self.parametres['b' + str(l)] = np.zeros((architecture[l], 1))

    # FONCTIONS D'ACTIVATION (De zéro)

    def _relu(self, Z):
        """Fonction d'activation ReLU (Rectified Linear Unit)."""
        # La fonction ReLU est définie comme f(x) = max(0, x). Elle introduit de la non-linéarité dans le modèle, ce qui permet de résoudre des problèmes complexes.
        # En effet, sans une fonction d'activation non linéaire, un réseau de neurones serait simplement une combinaison linéaire de ses entrées, ce qui limite sa capacité à modéliser des relations complexes dans les données.
        return np.maximum(0, Z)

    def _relu_derivee(self, Z):
        """Dérivée de la fonction ReLU pour la rétropropagation."""
        # La dérivée de ReLU est 1 pour les valeurs positives et 0 pour les valeurs négatives, ce qui permet de propager les gradients uniquement à travers les neurones activés (ceux qui ont une sortie positive).
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        """
        Fonction d'activation Sigmoid.
        """
        # La fonction Sigmoid est définie comme f(x) = 1 / (1 + exp(-x)). Elle est utilisée pour la couche de sortie dans les problèmes de classification binaire, car elle produit des valeurs entre 0 et 1, interprétables comme des probabilités.
        # Si on n'avait pas utilisé Sigmoid à la sortie, on aurait des résultats pour pourraient ressembler à A : 0.122, B : 0.212, difficile à interpréter, alors qu'avec Sigmoid on aurait A : 0.03, B : 0.97, ce qui est plus clair pour savoir quelle classe est prédite.
        # On utilise np.clip pour éviter les problèmes de débordement numérique (overflow) qui peuvent survenir avec la fonction exponentielle lorsque les valeurs de Z sont très grandes ou très petites. En limitant les valeurs de Z à une plage raisonnable, on garantit que la fonction Sigmoid reste stable et ne produit pas de résultats infinis ou NaN.
        Z_clipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_clipped))

    # LOGIQUE PRINCIPALE DU RESEAU

    def _propagation_avant(self, X):
        """
        Effectue une passe vers l'avant (Forward propagation) dans le réseau.
        
        Il prends en paramètre les données d'entrée X, et retourne les prédictions finales A ainsi que les caches nécessaires pour la rétropropagation.
        Les données sont sous la forme d'une matrice de dimension (nb_features, nb_exemples) pour faciliter les calculs matriciels dans le réseau.
        nb_features : nombre de caractéristiques (features) dans les données d'entrée
        nb_exemples : nombre d'exemples dans les données d'entrée
        Les prédictions finales A sont de dimension (1, nb_exemples) car nous avons une seule sortie pour chaque exemple (classification binaire).
        """

        # On utilise un dictionnaire 'caches' pour stocker les valeurs intermédiaires de Z et A à chaque couche, ce qui sera nécessaire pour la rétropropagation.
        caches = {'A0': X}
        A = X
        
        # Propagation à travers les couches cachées (activation ReLU)
        # Pour chaque couche l de 1 à L-1, on calcule Z = W * A + b (on applique le poids et le biais), puis on applique la fonction d'activation ReLU pour obtenir A. On stocke ensuite Z et A dans le cache pour la rétropropagation.
        for l in range(1, self.L):
            W = self.parametres['W' + str(l)]
            b = self.parametres['b' + str(l)]
            
            Z = np.dot(W, A) + b
            A = self._relu(Z)
            
            # On les stock pour la retropropagation
            caches['Z' + str(l)] = Z
            caches['A' + str(l)] = A
            
        # Propagation dans la dernière couche (activation Sigmoid)
        # Comme expliqué plus haut, cela permets d'avoir des prédictions entre 0 et 1, interprétables comme des probabilités pour la classification binaire.
        W = self.parametres['W' + str(self.L)]
        b = self.parametres['b' + str(self.L)]
        
        Z = np.dot(W, A) + b
        A = self._sigmoid(Z) # Output final entre 0 et 1
        
        # On les stock aussi pour la rétropropagation
        caches['Z' + str(self.L)] = Z
        caches['A' + str(self.L)] = A
        
        # On renvoie les prédictions finales A et les caches pour la rétropropagation
        return A, caches

    def _calculer_cout(self, A, Y):
        """
        Lorsque pour la sortie on utilise l'activation Sigmoid, il faut utiliser l'entropie croisée binaire (Binary Cross Entropy) comme fonction de coût, car elle est spécialement conçue pour ce type problèmes.
        """
        m = Y.shape[1] # Nombre d'exemples
        epsilon = 1e-15 # log(0) est indéfini, donc quand on utilise log, on ajoute une valeur très proche de zéro (epsilon) pour éviter les problèmes de calcul numérique.
        
        # Formule mathématique de la log loss (Entropie croisée binaire)
        cout = (-1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

        # On utilise np.squeeze pour transformer le résultat en un scalaire (au lieu d'un tableau de dimension (1,)) pour faciliter l'affichage et les calculs ultérieurs.
        return np.squeeze(cout)

    def _retropropagation(self, Y, caches):
        """
        Calcule les gradients via la rétropropagation (Backward propagation).

        On prends alors en entrée les vraies étiquettes Y et les caches contenant les valeurs de Z et A de la propagation avant.
        On renvoie un dictionnaire de gradients pour chaque couche, qui seront utilisés pour mettre à jour les poids et les biais du réseau.
        """
        m = Y.shape[1]
        gradients = {}
        
        # 1. Gradient de la dernière couche (Sigmoid + Binary Cross Entropy)
        # La dérivée de l'erreur par rapport à Z pour Sigmoid+BCE se simplifie tout simplement en (A - Y).
        A_final = caches['A' + str(self.L)]
        dZ = A_final - Y
        
        # Calcul des gradients pour les poids et les biais de la dernière couche
        A_prev = caches['A' + str(self.L - 1)]
        gradients['dW' + str(self.L)] = (1 / m) * np.dot(dZ, A_prev.T)
        gradients['db' + str(self.L)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # 2. Gradients pour les couches cachées (de L-1 jusqu'à 1) avec ReLU
        # Ici, on utilise la règle de la chaîne qui est définie comme ceci :
        # dA = W_next.T * dZ (propagation du gradient à travers les poids de la couche suivante)
        # dZ = dA * ReLU'(Z_current) (application de la dérivée de la fonction d'activation ReLU pour obtenir le gradient par rapport à Z)

        for l in reversed(range(1, self.L)):
            W_next = self.parametres['W' + str(l + 1)]
            Z_current = caches['Z' + str(l)]
            A_prev = caches['A' + str(l - 1)]
            
            # Application de la règle de la chaîne (Chain rule)
            dA = np.dot(W_next.T, dZ)
            dZ = dA * self._relu_derivee(Z_current)
            
            # Calcul des gradients pour les poids et les biais de la couche actuelle
            gradients['dW' + str(l)] = (1 / m) * np.dot(dZ, A_prev.T)
            gradients['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
        return gradients

    def _mettre_a_jour_parametres(self, gradients, taux_apprentissage):
        """
        Mise à jour des poids et biais avec l'algorithme de descente de gradient.
        """
        # Pour chaque couche l, on met à jour les poids W et les biais b en soustrayant le produit du taux d'apprentissage et des gradients correspondants.
        for l in range(1, self.L + 1):
            self.parametres['W' + str(l)] -= taux_apprentissage * gradients['dW' + str(l)]
            self.parametres['b' + str(l)] -= taux_apprentissage * gradients['db' + str(l)]

    # METHODES PUBLIQUES (API de l'utilisateur)
    # Ce que l'utilisateur va utiliser comme si c'était une bibliothèque avancée, sans se soucier de la logique interne du réseau de neurones.

    def entrainer(self, X, Y, epochs=1000, taux_apprentissage=0.01, afficher_cout=True):
        """
        Entraîne le réseau de neurones.

        Elle prends en paramêtres :
        - X : les données d'entraînement (dimension: nb_exemples, nb_features)
        - Y : les labels correspondants (dimension: nb_exemples, 1)
        - epochs : le nombre d'itérations sur l'ensemble des données
        - taux_apprentissage : le pas d'apprentissage pour la descente de gradient (learning rate)
        - afficher_cout : booléen pour afficher le coût (loss) tous les 1000 epochs (pour suivre la progression de l'entraînement)
        """
        # On transpose X et Y en (nb_features, nb_exemples) et (1, nb_exemples) respectivement pour faciliter les calculs matriciels dans le réseau de neurones.
        X_T = X.T 
        Y_T = Y.T.reshape(1, -1)
        
        # On effectue l'entraînement en boucle pour le nombre d'epochs spécifié. (cela premets d'avoir un réseau performant même si on a un petit dataset)
        # Il y a quatre étapes principales dans chaque epoch :
        for epoch in range(epochs):
            # 1. Propagation avant pour obtenir les prédictions et les caches nécessaires pour la rétropropagation
            A, caches = self._propagation_avant(X_T)
            
            # 2. Calcul du coût (loss) pour évaluer la performance du modèle pour l'epoch actuel
            cout = self._calculer_cout(A, Y_T)
            
            # 3. Rétropropagation pour calculer les gradients nécessaires à la mise à jour des poids et biais du réseau
            gradients = self._retropropagation(Y_T, caches)
            
            # 4. Mise à jour des paramètres du réseau (poids et biais) en utilisant les gradients calculés et le taux d'apprentissage spécifié
            self._mettre_a_jour_parametres(gradients, taux_apprentissage)
            
            # Si afficher_cout est True, on affiche le coût (loss) tous les 1000 epochs pour suivre la progression de l'entraînement. Cela permet de voir si le modèle s'améliore au fil du temps.
            # On peut changer la fréquence d'affichage en modifiant la condition epoch % 1000 == 0, par exemple pour afficher tous les 500 epochs, on mettrait epoch % 500 == 0.
            if afficher_cout and epoch % 1000 == 0:
                print(f"Epoch {epoch} | Coût (Loss): {cout:.4f}")

    def predire(self, X):
        """
        Prédit l'étiquette (0 ou 1) pour de nouvelles données.
        
        On prends en paramètre les données d'entrée X (dimension: nb_exemples, nb_features) et on retourne les prédictions (dimension: nb_exemples, 1) où chaque valeur est 0 ou 1.
        """
        X_T = X.T
        probabilites, _ = self._propagation_avant(X_T)
        
        # On transforme la probabilité en classe (1 si > 0.5, sinon 0) car on veux que la réponse soit un booléen
        # On peut cependant renvoyer les probabilités brutes si on veut un résultat plus détaillé
        predictions = (probabilites > 0.5).astype(int)
        
        # On transpose le résultat pour retourner à la dimension (nb_exemples, 1)
        return predictions.T

# TEST DU RESEAU DE NEURONES

# Ici, on entraîne notre réseau sur le problème du XOR car il ne peut pas être résolu par un perceptron simple, cela signifie qu'il va nous faloir un réseau de neurones multicouches.
# On prendra l'architecture suivante : [2, 8, 4, 1] :
# 2 car nous avons 2 features en entrée (les deux bits du XOR)
# 8 et 4 suffisent pour résoudre le problème, il faudrait plus de couches cachées avec plus de neurones pour des problèmes plus complexes, mais pour le XOR cela suffit largement.
# 1 car nous avons une seule sortie (0 ou 1 pour le XOR)
# On prendra un nombre d'epochs élevé (5000) pour s'assurer que le réseau converge vers une solution optimale, et un taux d'apprentissage de 0.05 pour permettre des mises à jour significatives des poids et biais à chaque itération, ce qui aide à accélérer l'entraînement tout en évitant les oscillations dans la convergence.

if __name__ == "__main__":
    # On créé le dataset
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # On créé les étiquettes
    Y_train = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    
    # On initialise le réseau de neurones avec l'architecture expliquée plus haut
    architecture = [2, 8, 4, 1]
    mon_reseau = ReseauNeurones(architecture)
    
    # On choisit des valeurs pour le nombre d'epochs et le taux d'apprentissage comme détaillé plus haut
    epochs = 5000
    taux_apprentissage = 0.05
    
    # On appelle notre méthode d'entraînement pour entraîner le réseau de neurones sur notre dataset (avec afficher_cout à True pour savoir si on a mis un nombre d'epochs trop haut)
    mon_reseau.entrainer(X_train, Y_train, epochs=epochs, taux_apprentissage=taux_apprentissage, afficher_cout=True)
    
    # On test notre modele sur toutes les valeurs possibles
    predictions = mon_reseau.predire(X_train)
    
    # On calcule la precision moyenne (sur un problème aussi simple, si on n'a pas 100%, c'est que notre réseau est absolument nul)
    # (Remarque : on affiche un floatant mais ici les valeurs possibles ne sont que des valeurs entières, j'ai mis ca au cas où vous voudriez tester sur un autre exemple)
    precision = np.mean(predictions == Y_train) * 100
    print(f"\nPrécision du modèle : {precision:.2f}%")
    
    # Affichage détaillé des résultats
    print("\n\nDétail :")
    for i in range(len(X_train)):
        entree = X_train[i]
        attendu = Y_train[i][0]
        predit = predictions[i][0]
        succes = "V" if attendu == predit else "X"
        print(f"{entree} -> {predit} {succes}")
