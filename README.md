# Deep Convolutional Generative Adversarial Network (DCGAN) sur MNIST

## Aperçu

Ce projet implémente un **Deep Convolutional Generative Adversarial Network (DCGAN)** en utilisant TensorFlow/Keras pour générer de nouvelles images de chiffres manuscrits èa partir de l'ensemble **MNIST**).

L'objectif principal est de définir et d'entraîner deux réseaux neuronaux :
1.  **Le Générateur (G) :** Apprend à générer des images des chiffres à partir d'un vecteur de bruit aléatoire.
2.  **Le Discriminateur (D) :** Apprend à distinguer les vraies images du dataset des fausses images générées par le Générateur.

## Architecture du Modèle

Le modèle est construit selon les principes des DCGAN, en utilisant des couches de convolution pour l'ensemble du réseau.

### 1. Générateur (G)

Le Générateur prend un vecteur de bruit latent (dimension 256) et utilise des **Couches de Convolution Transposée** pour effectuer un sur-échantillonnage progressif de la taille de l'image.

* **Entrée :** Vecteur latent de dimension **256**.
* **Couches :** Dense $\rightarrow$ Reshape ($7 \times 7 \times 128$) $\rightarrow$ Couches `Conv2DTranspose` avec `BatchNormalization` et `LeakyReLU`.
* **Sortie :** Image $28 \times 28 \times 1$ avec activation `tanh` (valeurs entre -1 et 1).

### 2. Discriminateur (D)

Le Discriminateur prend une image ($28 \times 28 \times 1$) et utilise des **Couches Convolutionnelles (`Conv2D`)** pour effectuer un sous-échantillonnage progressif afin d'extraire les caractéristiques.

* **Entrée :** Image $28 \times 28 \times 1$.
* **Couches :** Couches `Conv2D` avec `LeakyReLU` $\rightarrow$ `Dropout` $\rightarrow$ `Flatten`.
* **Sortie :** Scalaire unique avec activation `sigmoid` (probabilité que l'image soit réelle).

## ⚙️ Configuration d'Entraînement

| Paramètre | Valeur |
| :--- | :--- |
| **Optimiseur (G et D)** | RMSprop |
| **Taux d'Apprentissage (G)** | $5 \times 10^{-4}$ |
| **Taux d'Apprentissage (D)** | $5 \times 10^{-5}$ |
| **Fonction de Perte** | Binary Cross-Entropy (BCE) |
| **Taille du Batch** | 64 |
| **Label Smoothing** | Oui (labels réels fixés à 0.9) |
| **Époques (Entraînement)** | 20 |

## Résultats

Les images ci-dessous illustrent comment la qualité des images générées s'améliore au fil des époques, passant de formes abstraites à des chiffres reconnaissables.

| Époque 1 | Époque 5 | Époque 18 |
| :---: | :---: | :---: |
| <img width="320" height="240" alt="generated_random_images_epoch_1" src="https://github.com/user-attachments/assets/04828281-3c35-42c8-b2ce-ca657a93885e" /> | <img width="320" height="240" alt="generated_random_images_epoch_10" src="https://github.com/user-attachments/assets/1fbce482-fe3e-4edc-8fed-76d94b90b971" /> | <img width="320" height="240" alt="generated_random_images_epoch_20" src="https://github.com/user-attachments/assets/4efc171d-846a-411c-9721-5cb97d192ca1" /> |

### Performance des Pertes (Loss)

Les pertes du Discriminateur (D Loss) et du Générateur (GAN Total Loss) après **20 époques** montrent que l'entraînement est stable et que le Générateur apprend à tromper le Discriminateur :

* **D Loss (Discriminateur)**: **0.6619** (Proche de 0.693, la valeur idéale pour un équilibre parfait, indiquant une bonne stabilité.)
* **GAN Total Loss (Générateur)**: **0.0851** (Une valeur faible pour le Générateur indique qu'il réussit de plus en plus à faire en sorte que le Discriminateur attribue une haute probabilité (proche de 1) aux fausses images, car la perte est calculée sur une cible de 1.)

<img width="320" height="240" alt="Loss" src="https://github.com/user-attachments/assets/40f99d56-5f5e-4811-a0d3-f2c12934f4c3" />

## Limites et Pistes d'Amélioration

Bien que le modèle produise des chiffres globalement reconnaissables, la qualité n'est pas encore photoréaliste pour certains échantillons, et la diversité pourrait être améliorée.
Pour une amélioration significative de la qualité et de la diversité des images, plusieurs pistes pourraient être explorées :

1.  **Augmentation de la Complexité du Modèle :**
    * **Augmenter le nombre de paramètres** du Générateur (par exemple, en augmentant la taille du vecteur latent ou le nombre de filtres dans les couches convolutionnelles) pour lui permettre d'apprendre des caractéristiques plus fines.
2.  **Ajustement des Hyperparamètres :**
    * Poursuivre l'entraînement pour un **plus grand nombre d'époques** (ex. : 100-200) pour affiner la convergence.
    * Modifier le ratio d'entraînement entre le Discriminateur et le Générateur (par exemple, entraîner le Discriminateur 2 ou 3 fois pour chaque mise à jour du Générateur) afin de maintenir un meilleur équilibre.
3.  **Architecture Avancée :**
    * Remplacer la perte Binary Cross-Entropy (BCE) par une perte **Wasserstein GAN (WGAN)** avec pénalité de gradient (WGAN-GP), reconnue pour améliorer la stabilité de l'entraînement des GANs et la qualité des échantillons générés.
