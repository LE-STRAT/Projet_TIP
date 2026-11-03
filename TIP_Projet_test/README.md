# Food-11 : classification d'images par transfert d'apprentissage (MATLAB)

Projet de Traitement d'Images et Paroles (TIP). Classification de 11 catégories d'aliments (jeu de données Food-11) à partir d'un réseau pré-entraîné (ResNet-18 par défaut) dont la dernière couche est remplacée.

## Contenu

Un seul script MATLAB, découpé en sections :

1. **Données** : lecture du dossier `train` (sous-dossiers = classes), découpage 80 % entraînement / 20 % validation.
2. **Réseau** : réseau pré-entraîné, remplacement de la couche finale (11 classes), calcul du nombre de paramètres.
3. **Entraînement** : `trainnet` avec l'optimiseur Adam.
4. **Évaluation** : F1 macro, score de frugalité, matrice de confusion (sur la validation).
5. **Prédictions** : génération de `predictions.json` sur le jeu `test` (à lancer seulement avec le meilleur modèle).

## Prérequis

- MATLAB R2024a ou plus récent
- Deep Learning Toolbox
- Le support package du réseau choisi (ex. *Deep Learning Toolbox Model for ResNet-18 Network*), installable via **Add-Ons**
- Le jeu de données Food-11 dézippé, avec les dossiers `train/` et `test/`

## Utilisation

1. Modifier `trainDir` et `testDir` (section 1) pour pointer vers vos dossiers. Les chemins du script sont absolus et propres à ma machine.
2. Ajuster au besoin `netName`, `inputSize` et `trainingOptions` (voir ci-dessous).
3. Exécuter les sections 1 à 4 pour entraîner et évaluer.
4. Exécuter la section 5 uniquement avec le meilleur modèle, pour produire `predictions.json` (format `{"nom_image": "classe"}`).

## Contexte d'exécution des résultats

L'entraînement a été réalisé **sans carte graphique (CPU uniquement)**. Avec la configuration par défaut du script (ResNet-18, images 128×128, batch de 64, 5 epochs), il a duré **environ 1 heure**. Les résultats sont donc ceux d'une configuration volontairement légère. Sur GPU, un budget d'entraînement plus large est possible et devrait améliorer le F1 (voir tableau).

## Résultats (configuration CPU : ResNet-18, 128×128, batch 64, 5 epochs)

| Mesure | Valeur |
|---|---|
| Paramètres | 11,19 M |
| F1 macro (validation) | 0,7201 |
| Frugalité | 0,4447 |
| Exactitude (validation) | ≈ 72 % (1925 / 2659 images) |

**Analyse de la matrice de confusion** (rappel par classe, sur la validation) :

- **Bien reconnues** : soup (≈ 93 %), vegetable-fruit (≈ 89 %), rice (≈ 85 %), seafood (≈ 84 %), meat (≈ 83 %), egg (≈ 80 %).
- **Faibles** : bread (≈ 42 %), fried (≈ 49 %), dairy (≈ 58 %), dessert (≈ 59 %).
- **Confusions principales** : bread → egg (55) et meat (32) ; fried → meat (41) ; dessert → seafood (47), meat (36) et soup (28).
- **Classes « aimants »** : les prédictions meat (précision ≈ 66 %) et seafood (≈ 56 %) absorbent beaucoup de faux positifs.
- **Déséquilibre** : rice (75 images) et noodles-pasta (117) sont rares face à soup (400) et meat (355).

**Pistes d'amélioration** (hypothèses non vérifiées) : bread, fried et dessert regroupent des plats visuellement variés, donc une résolution plus élevée (224×224), plus d'epochs et l'augmentation de données devraient aider surtout ces classes. Sans GPU, un réseau plus léger (`squeezenet`, ≈ 1,2 M de paramètres) améliorerait la frugalité, à condition que le F1 ne baisse pas trop.

## Hyperparamètres à modifier selon le matériel

Les valeurs « GPU » sont des points de départ indicatifs à valider par essais. Je ne les ai pas mesurées, ma machine n'ayant pas de GPU.

| Paramètre | Sans GPU (config utilisée) | Avec GPU (proposition) | Facteur / remarque |
|---|---|---|---|
| `inputSize` | `[128 128 3]` (`[96 96 3]` pour des essais rapides) | `[224 224 3]` | ×1,75 par côté, soit ≈ ×3 en pixels, donc ≈ ×3 de calcul par image |
| `MaxEpochs` | 3 à 5 | 10 à 20 | ×2 à ×4 |
| `MiniBatchSize` | 32 à 64 | 64 à 128 | ×2 si la mémoire GPU le permet, sinon rester à 64 |
| `InitialLearnRate` | `1e-3` | `1e-4` à `3e-4` | ÷3 à ÷10 pour un réglage plus fin sur davantage d'epochs |
| `LearnRateSchedule` | `"none"` (par défaut) | `"piecewise"` (`LearnRateDropFactor=0.1`, `LearnRateDropPeriod=5`) | Utile dès que le nombre d'epochs augmente |
| `ValidationFrequency` | 300 | ≈ 1 epoch d'itérations | Itérations par epoch = nb images d'entraînement / `MiniBatchSize` ; à recalculer si le batch change |
| Augmentation (`imageDataAugmenter`) | Désactivée (coûteuse en CPU) | `RandXReflection=true`, `RandRotation=[-15 15]` | À activer quand `MaxEpochs` ≥ 10, pour limiter le surapprentissage |
| `ExecutionEnvironment` | `"cpu"` | `"gpu"` | `"auto"` (défaut) choisit le GPU s'il est disponible |
| `netName` | `"resnet18"` ou plus léger (`"squeezenet"`, `"mobilenetv2"`) | `"resnet18"`, ou `"resnet50"` si la frugalité le permet | Un réseau plus gros augmente `nParams`, donc pénalise le score de frugalité |

**Ordre de grandeur du coût** : le temps d'entraînement évolue à peu près comme `MaxEpochs × (largeur × hauteur)`. Passer de 128 à 224 de côté multiplie le coût par ≈ 3 : sur CPU, 224 px avec 5 epochs prendrait plusieurs heures. À éviter sans GPU.

**Conseil de démarche** : faire varier un seul paramètre à la fois (taille d'entrée, puis epochs, puis learning rate, puis augmentation) et noter le F1 macro et la frugalité à chaque essai.

## Métriques

- **F1 macro** : moyenne des F1 par classe, calculée à partir de la matrice de confusion.
- **Frugalité** : `F1 − 0.3 / (1 + 1/R)`, avec `R` = nombre de paramètres en millions. Un petit réseau est donc favorisé à F1 égal.
- **Matrice de confusion** : affichée via `confusionchart`, utile pour l'analyse des erreurs dans le rapport.

## Sorties

- Courbes d'entraînement (fenêtre `training-progress`)
- F1 macro et frugalité dans la console
- Matrice de confusion
- `predictions.json` (prédictions sur le jeu de test)
