# Prédiction de Prix de Voitures Audi (Deep Learning)

Ce projet implémente un pipeline complet de Deep Learning utilisant **TensorFlow/Keras** pour prédire le prix de vente de véhicules Audi en fonction de leurs caractéristiques (kilométrage, modèle, carburant, etc.).

## 📌 Description

L'objectif est de résoudre un problème de **régression** en entraînant un réseau de neurones artificiels (ANN). Le flux de travail inclut l'analyse exploratoire, un prétraitement des données rigoureux, et l'entraînement d'un modèle optimisé.

Les étapes clés du projet sont :
1.  **Exploration des données (EDA)** : Visualisation des relations (ex: Prix vs Distance).
2.  **Prétraitement (Preprocessing)** :
    *   Encodage des variables catégorielles (`LabelEncoder` et `TargetEncoder`).
    *   Normalisation des données numériques (`StandardScaler`).
    *   Division des données en ensembles d'Entraînement (70%), Validation (15%) et Test (15%).
3.  **Modélisation** : Réseau de neurones séquentiel avec couches denses et régularisation (Dropout).
4.  **Sauvegarde** : Export du modèle (`.keras`) et des pipelines de transformation (`.pkl`) pour une réutilisation future.

## 🛠 Technologies et Bibliothèques

*   **Python**
*   **Deep Learning** : TensorFlow, Keras
*   **Data Science** : Pandas, NumPy, Scikit-Learn
*   **Visualisation** : Matplotlib, Seaborn
*   **Outils Spécifiques** : `category_encoders` (Target Encoding), `joblib` (Sauvegarde)

## 🧠 Architecture du Modèle

Le modèle utilisé est un Perceptron Multicouche (MLP) avec la structure suivante :
*   **Entrée** : Caractéristiques prétraitées du véhicule.
*   **Couches Cachées** :
    *   Dense (512 neurones, ReLU) + Dropout (30%)
    *   Dense (256 neurones, ReLU) + Dropout (20%)
    *   Dense (128 neurones, ReLU)
*   **Sortie** : 1 neurone (Activation linéaire) pour prédire le prix continu.
*   **Optimiseur** : Adam
*   **Fonction de perte** : Mean Squared Error (MSE)

## 🚀 Installation et Utilisation

1.  **Prérequis** : Assurez-vous d'avoir Python installé avec les bibliothèques suivantes :
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow category_encoders joblib openpyxl
    ```

2.  **Données** : Le projet attend un fichier de données nommé `Audi.xls`. Modifiez le chemin d'accès dans le notebook si nécessaire (`df = pd.read_excel(...)`).

3.  **Exécution** : Lancez le notebook `Untitled0.ipynb` pour :
    *   Charger et préparer les données.
    *   Entraîner le modèle.
    *   Visualiser les performances (Loss, RMSE).
    *   Obtenir le score R² sur les données de test.

## 📊 Résultats et Métriques

Le modèle est évalué en utilisant :
*   **RMSE (Root Mean Squared Error)** : Pour quantifier l'écart moyen entre le prix prédit et le prix réel.
*   **R² Score** : Pour évaluer la qualité de l'ajustement du modèle sur les données de test.

Des mécanismes de **Early Stopping** et de **Réduction du Learning Rate** sont intégrés pour éviter le surapprentissage (overfitting) et optimiser la convergence.
