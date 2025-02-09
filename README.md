# Projets XAI - README & Rapport

**Elyes KHALFALLAH** & **Edouard CHAPPON**

10/02/2025

---

## Preface

### Exécution du Projet

Chaque projet a son propre code en `projetN.ipynb`.

### Installation et Lancement de l’Environnement Virtuel

1. **Créer l'environnement virtuel :**

   ```bash
   python -m venv env_xai
   ```

2. **Activer l'environnement virtuel :**

   - **Sous Windows**

     ```bash
     .\env_xai\Scripts\activate
     ```

   - **Sous macOS et Linux**

     ```bash
     source env_xai/bin/activate
     ```

3. **Installer les dépendances :**

   ```bash
   pip install -r requirements.txt
   ```
   Il est important de noter que certaines dépendances (particulierement `dice_ml`) sont particulièrement difficiles à faire fonctionner correctement... Pour cela, elles ne sont pas incluses dans les requirements. 
---

## Rapport

---

### **Projet 1. SHAPLEY Values**

Ce projet explore l'explicabilité des modèles de machine learning à travers l'utilisation de SHAP (SHapley Additive exPlanations) appliqué à un classificateur XGBoost. L'objectif est de comprendre et d'interpréter les décisions d'un modèle entraîné sur le dataset Adult Income de l'UCI Machine Learning Repository.

Ce dataset contient des informations démographiques et professionnelles sur des individus, avec pour objectif de prédire si leur revenu annuel dépasse 50 000 dollars par an (income : >50K ou <=50K). Il comprend plusieurs caractéristiques telles que l'âge, l'éducation, le statut marital ou encore le nombre d'heures travaillées par semaine.

Un pretraitement de donnees est effectué, le modèle choisi pour la classification est un XGBClassifier, un algorithme basé sur des arbres de décision optimisé pour des performances élevées. Les principales étapes sont :

- Division des données en ensemble d'entraînement (70%) et de test (30%), en respectant la distribution de la variable cible.
- Ajustement du paramètre scale_pos_weight pour gérer le déséquilibre des classes.
- Entraînement du modèle avec 200 arbres (n_estimators=200), une profondeur maximale de 8 (max_depth=8) et un taux d'apprentissage de 0.05.

L'interprétation du modèle est réalisée avec SHAP, qui permet :

- De visualiser l'impact de chaque feature sur les prédictions globales via un summary_plot.
- D'expliquer les décisions du modèle pour des individus spécifiques grâce à des force_plot, illustrant les contributions des caractéristiques.

On y voit que les statistiques marital-status et age sont parmi les plus decisives.

---

### **Projet 2. Counterfactuals pour données tabulaires**

L'objectif de ce projet est d'explorer l'explicabilité d'un modèle de classification appliqué au jeu de données Breast Cancer Wisconsin provenant de scikit-learn. Nous utilisons un classificateur Random Forest pour prédire la présence d'un cancer et analysons son comportement à l'aide de SHAP et DiCE pour mieux comprendre les prédictions et les contrefactuels.

Le dataset contient des mesures issues d'analyses de tumeurs du sein, avec des caractéristiques décrivant leur texture, leur périmètre ou encore leur concavité. La variable cible target est binaire et prend la valeur 1 pour les cas de cancer et 0 pour les cas sains.

Comme l'exemple donné, random forest est le modèle utilisé.

Nous utilisons DiCE (Diverse Counterfactual Explanations) pour ensuite générer des contrefactuels, c'est-à-dire des exemples proches des observations réelles mais avec une classification inverse. L'objectif est de comprendre quels changements dans les caractéristiques pourraient modifier la décision du modèle.

Ce projet illustre comment l'explicabilité permet de mieux comprendre les décisions d'un modèle de machine learning. L'utilisation combinée de matrices de confusion, DiCE et l'analyse des prédictions peu confiantes offre une vision plus approfondie du comportement du classificateur Random Forest. Ces méthodes peuvent être essentielles dans des domaines critiques comme la santé, où l'interprétabilité des modèles est primordiale.

---

### **Projet 3. Grad Cam pour analyse d'images**

L'objectif de ce projet est d'explorer l'explicabilité d'un modèle de deep learning en analysant ses zones d'attention lors de la classification d'images. Nous utilisons le modèle pré-entraîné Xception et la technique Grad-CAM pour visualiser les activations des couches convolutives.

Le modèle utilisé est Xception, un réseau de neurones convolutifs avancé pré-entraîné sur ImageNet. L'image analysée est une photo d'une souris de maison trouvée chez Elyes.

La méthode Grad-CAM (Gradient-weighted Class Activation Mapping) permet d'obtenir une carte de chaleur indiquant les régions les plus importantes pour la prédiction d'une classe.

1. Passage de l'image dans le modèle : L'image est redimensionnée et prétraitée pour correspondre aux entrées du modèle.

2. Extraction des activations : On isole les activations de la dernière couche convolutive.

3. Calcul des gradients : On dérive les activations par rapport à la classe prédite.

4. Pondération des activations : Les activations sont multipliées par l'intensité de leurs gradients.

5. Génération de la heatmap : La carte de chaleur obtenue est superposée sur l'image d'origine.

Le modèle a prédit la classe "mousetrap" avec une probabilité relativement élevée. La visualisation de Grad-CAM montre que le modèle se focalise principalement sur le corps, les yeux, et les oreilles de l'animal, ce qui est cohérent avec la manière dont les humains identifient une souris.

---

### **Projet 4. Interpretation de BERT**

Ce projet vise à analyser l'interprétabilité du modèle BERT en utilisant Captum, une bibliothèque d'explicabilité pour les modèles de deep learning basés sur PyTorch. L'objectif est d'examiner comment BERT détermine ses réponses dans une tâche de question-réponse, en mettant en évidence les tokens les plus influents dans la décision du modèle.

Le modèle utilisé est BERT-base-uncased, un modèle pré-entraîné de la bibliothèque Transformers de Hugging Face. Il est spécifiquement employé pour la tâche de question-réponse avec BertForQuestionAnswering.

Les données sont constituées d'une question et d'un texte :

- Question : "Why do people take medication?"

- Texte : "Medication is necessary for relieving illnesses and saving lives."

Les tokens sont générés via le tokenizer BERT et les tenseurs nécessaires sont construits, incluant les IDs des tokens, les masques d'attention et les IDs de type de token.

Le modèle effectue une prédiction des scores de début et de fin de la réponse. La réponse prédite est extraite en fonction des scores maximaux.

Pour interpréter la décision du modèle, Layer Integrated Gradients (LIG) est utilisé afin d'identifier les tokens qui ont le plus influencé la réponse prédite.

Les résultats montrent que BERT essaye d'identifier les mots clés les plus pertinents dans le texte pour déterminer sa réponse. L'analyse des scores d'attribution permet de vérifier si le modèle prend ses décisions sur des bases cohérentes ou s'il est sujet à des biais.

Cette approche est essentielle pour l'interprétabilité des modèles de deep learning, en particulier dans des applications critiques comme la médecine ou le droit.

---

### **Projet 5. Layerwise Relevance Propagation pour l'analyse d'image**

Ce projet implémente un réseau de neurones simple avec propagation avant (feedforward), suivi d'une méthode de calcul de la "relevance" de chaque neurone dans les différentes couches du réseau. Le code applique des poids aléatoires et les valeurs d'entrée définies par l'utilisateur, puis utilise des équations pour obtenir une sortie, avant de calculer la "relevance" des neurones en utilisant des méthodes de propagation de la "relevance" (LRP). Enfin, il vérifie la positivité et la conservativité des "relevances" via des assertions et des tests unitaires.

Le code commence par définir un réseau de neurones simple avec trois couches : une couche d'entrée (i), une couche cachée (j), et une couche de sortie (k). Les valeurs des entrées et des poids entre les couches sont spécifiées manuellement ou générées aléatoirement. Un histogramme des poids de la couche de sortie (k) est affiché pour observer leur distribution.

Le réseau effectue ensuite un calcul de propagation avant (feedforward) en utilisant des équations de somme pondérée pour chaque neurone de la couche cachée. La fonction ReLU est ensuite appliquée sur les sorties de la couche cachée, et un calcul final donne la sortie de la couche de sortie (k).

La méthode de propagation des relevances (LRP) est appliquée pour calculer l'importance de chaque neurone dans les couches cachées et d'entrée. La pertinence des neurones dans la couche de sortie est égale à la sortie du réseau, et celle des neurones dans les couches cachées et d'entrée est calculée à l'aide des poids au carré normalisés.

Les propriétés de positivité (les relevances doivent être positives) et de conservativité (la somme des relevances d'une couche doit être égale à celle de la couche suivante) sont vérifiées par des assertions et des tests unitaires. Les résultats sont validés par un test de conformité des relations entre les couches.

Le code implémente un réseau de neurones simple avec un mécanisme de calcul des relevances basé sur la méthode LRP, tout en vérifiant les propriétés essentielles de ce calcul pour garantir la validité des résultats.

---

### **Projet 6. Layerwise Relevance Propagation pour réseaux de neurones graphiques**

Ce rapport présente l'analyse d'une implémentation Python d'un réseau de neurones sur graphes (GNN) conçu pour analyser et classifier des structures de graphes. Le système intègre également une méthode d'explication des prédictions basée sur la technique Layer-wise Relevance Propagation (LRP), permettant une meilleure compréhension des décisions du modèle.

Le projet s'articule autour de la génération de graphes scale-free suivant le modèle de Barabási-Albert, une approche qui produit des réseaux dont la distribution des degrés suit une loi de puissance. Ces graphes sont caractérisés par un paramètre de croissance qui détermine leur évolution structurelle au fil du temps.

L'architecture neuronale développée, implémentée dans la classe GraphNet, adopte une structure en quatre couches : une couche d'entrée, deux couches cachées, et une couche de sortie. Chaque couche utilise la fonction d'activation ReLU pour introduire de la non-linéarité dans le modèle. Cette architecture traite directement les matrices d'adjacence des graphes comme données d'entrée. L'entraînement du modèle s'effectue sur 20 000 itérations, utilisant l'optimiseur SGD (Stochastic Gradient Descent) avec momentum.

Cette implémentation représente une contribution significative à l'analyse explicable des réseaux de neurones sur graphes. Elle combine avec succès des techniques d'apprentissage profond avancées avec des méthodes d'explication sophistiquées, offrant ainsi un outil précieux pour l'analyse de structures de graphes complexes. Les possibilités d'amélioration incluent l'extension à des graphes de taille variable et l'optimisation automatique des hyperparamètres d'explication.
