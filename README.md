# Grad-ECLIP: Explainable CLIP via Gradient-based Attention Analysis

Ce projet reproduit et implémente la méthode **Grad-ECLIP** décrite dans l'article scientifique 2502.18816v1.pdf, qui propose une approche novatrice pour expliquer les décisions du modèle CLIP (Contrastive Language-Image Pre-Training) en utilisant les gradients des couches d'attention.

**🔗 Repository officiel :** Nous nous sommes inspirés du repository officiel de Grad-ECLIP : https://github.com/Cyang-Zhao/Grad-Eclip

##  Table des matières

- Vue d'ensemble
- Structure du projet
- Installation
- Téléchargement des données
- Notebooks principaux
- Méthodes d'explication comparées
- Évaluation quantitative
- Résultats
- Documentation
- Références

##  Vue d'ensemble

### Qu'est-ce que Grad-ECLIP ?

**Grad-ECLIP** est une méthode d'explicabilité pour les modèles vision-langage, spécifiquement conçue pour CLIP. Elle utilise les **gradients des couches d'attention** pour générer des cartes de saillance qui expliquent :

- **Pourquoi** une image correspond à un texte donné
- **Quelles parties** de l'image sont importantes pour cette correspondance
- **Comment** le modèle interprète la relation image-texte
- **Quels mots** du texte sont les plus pertinents pour l'image

### Avantages de Grad-ECLIP

-  **Simplicité** : Méthode directe basée sur les gradients
-  **Efficacité** : Pas de réentraînement nécessaire
-  **Polyvalence** : Applicable aux branches image ET texte
-  **Performance** : Surpasse les méthodes existantes sur les benchmarks

##  Structure du projet

```
├── 2502.18816v1.pdf                    #  Article scientifique de référence
├── rapport_projet_bgdia708_grad_clip.pdf  #  Rapport complet du projet
├── README.md                           #  Ce fichier
├── requirements.txt                    #  Dépendances Python
├── download_dataset.ipynb              #  Téléchargement automatique des datasets
├── valprep.sh                         #  Script d'organisation ImageNet
├── finetuning.md                      #  Documentation fine-tuning
├── imagenet_class_index.json          #  Index des classes ImageNet
├── imagenet_labels.txt                #  Labels ImageNet
├── whippet.png                        #  Image d'exemple
│
├── CLIP/                              #  Implementation CLIP originale
│   ├── clip/                          # Module CLIP core
│   ├── notebooks/                     # Notebooks d'exemple CLIP
│   └── requirements.txt               # Dépendances CLIP
│
├── Grad_CLIP/                         #  Notre implementation principale
│   ├── 📓 Notebooks d'explication
│   │   ├── grad_eclip_image.ipynb     # Explication image→texte
│   │   ├── grad_eclip_text.ipynb      # Explication texte→image
│   │   └── compare_visualize.ipynb    # Comparaison des méthodes
│   │
│   ├── 📊 Notebooks d'évaluation
│   │   ├── imagenet_eval_deletion.ipynb    # Test de suppression
│   │   ├── imagenet_eval_insertion.ipynb   # Test d'insertion
│   │   └── finetuning.ipynb               # Fine-tuning des modèles
│   │
│   ├── 🔧 Scripts utilitaires
│   │   ├── clip_utils.py               # Utilitaires CLIP
│   │   ├── generate_emap.py            # Génération des cartes d'explication
│   │   ├── imagenet_metadata.py        # Métadonnées ImageNet
│   │   ├── insertion_evaluation_results.csv # Résultats évaluation
│   │   └── test.py                     # Scripts de test
│   │
│   ├── 🎯 Méthodes comparées
│   │   ├── BLIP/                       # BLIP implementation
│   │   ├── CLIP_Surgery/               # CLIP Surgery method
│   │   ├── Game_MM_CLIP/              # GAME-MM method
│   │   └── M2IB/                      # M2IB method
│   │
│   ├── 📂 Données et résultats
│   │   ├── data/val/                   # Dataset de validation ImageNet
│   │   ├── images/                     # Images d'exemple
│   │   └── outfile/                    # Résultats de sortie
│   │
│   └── 🧪 Notebooks de développement
│       ├── pynvml_checkpoints/         # Points de contrôle
│       └── adaptation_vit.ipynb        # Adaptation Vision Transformer
│
├── concept_decomposition.png           #  Visualisation des concepts
├── map_comparaison.png                #  Comparaison des méthodes
├── textual_explanation.png            #  Explications textuelles
└── outfile/                           #  Résultats globaux du projet
```

## 🛠 Installation

### Prérequis

- **Python 3.8+**
- **CUDA 11.0+** (recommandé pour GPU)
- **Git**
- **Jupyter Notebook**

### Installation des dépendances

```bash
# Cloner le repository
git clone <repository-url>
cd Projet-IA-Fairness

# Installer les dépendances principales
pip install -r requirements.txt

# Installer CLIP
pip install -r CLIP/requirements.txt
```

## 🗂 Téléchargement des données

### Dataset principal : ImageNet

Le projet utilise **ImageNet ILSVRC2012** pour l'évaluation quantitative. Le notebook download_dataset.ipynb permet de télécharger automatiquement les données avec plusieurs options :

1. **Kaggle** (recommandé) - nécessite une API key
2. **Site officiel ImageNet** - nécessite inscription
3. **Academic Torrents** - plus fiable pour gros volumes
4. **Échantillon de test** - pour développement rapide

### Organisation des données

Après téléchargement, utilisez le script valprep.sh pour organiser le validation set ImageNet :

```bash
bash valprep.sh
```

## 📓 Notebooks principaux

### 1. `grad_eclip_image.ipynb` 
** Explication des images par le texte**

Implémente l'algorithme principal de Grad-ECLIP pour expliquer pourquoi une image correspond à un texte donné.

**Fonctionnalités :**
- Génération de cartes de saillance pour les images
- Visualisation des régions importantes avec heatmaps
- Superposition des explications sur l'image originale
- Export des résultats en haute résolution

### 2. `grad_eclip_text.ipynb`
** Explication du texte par l'image**

Implémente l'explication inverse : quels mots du texte sont importants pour la correspondance avec l'image.

**Fonctionnalités :**
- Attribution d'importance aux tokens textuels
- Visualisation des mots-clés avec codes couleur
- Analyse de la contribution de chaque mot
- Génération de nuages de mots pondérés

### 3. `compare_visualize.ipynb`
**🔍 Comparaison des méthodes d'explication**

Compare Grad-ECLIP avec les autres méthodes d'explicabilité disponibles sur les mêmes exemples.

**Comparaisons effectuées :**
- Visualisation côte-à-côte des différentes méthodes
- Métriques quantitatives de performance
- Analyse qualitative des explications

##  Méthodes d'explication comparées

### Grad-ECLIP (Notre méthode)
**Localisation :** generate_emap.py

- **Principe** : Utilise les gradients des couches d'attention de CLIP
- **Innovation** : Première méthode à exploiter spécifiquement les gradients d'attention cross-modale
- **Avantages** : Simple, efficace, interprétable

### BLIP 🔵
**Localisation :** BLIP

- **Principe** : Bootstrap vision-language understanding avec captioning
- **Architecture** : Vision Transformer + BERT multimodal

### CLIP Surgery 
**Localisation :** CLIP_Surgery

- **Principe** : Modification architecturale de CLIP pour améliorer la localisation
- **Méthode** : Remplace les couches d'attention par des versions "chirurgicales"

### GAME-MM 
**Localisation :** Game_MM_CLIP

- **Principe** : Gradient-weighted Class Activation Mapping pour le multimodal
- **Extension** : Adapte Grad-CAM aux modèles vision-langage

### M2IB 
**Localisation :** M2IB

- **Principe** : Multi-Modal Information Bottleneck
- **Théorie** : Minimise l'information mutuelle tout en préservant la performance

## 📊 Évaluation quantitative

### Tests de performance sur ImageNet

Le projet inclut deux notebooks d'évaluation quantitative sur les **1000 classes d'ImageNet** :

#### 1. `imagenet_eval_deletion.ipynb`
**Test de suppression** - Mesure la baisse de performance quand on supprime progressivement les régions importantes identifiées par chaque méthode.

**Métriques :**
- AUC (Area Under Curve) de suppression
- Fidélité des explications
- Pente de dégradation

#### 2. `imagenet_eval_insertion.ipynb` 
**Test d'insertion** - Mesure l'amélioration de performance quand on révèle progressivement les régions importantes sur une image masquée.

**Métriques :**
- Courbes d'insertion
- Efficacité (pourcentage minimum de pixels pour 90% de performance)
- Comparaison inter-méthodes

### Classes et templates ImageNet

Le projet utilise les **1000 classes standard d'ImageNet** avec **80 templates d'augmentation** optimisés pour CLIP, permettant une évaluation robuste sur l'ensemble du dataset.

## 📈 Résultats

### Performance comparative sur ImageNet

| Méthode | Fidélité | Localisation | AUC Deletion | AUC Insertion | Temps (ms) |
|---------|----------|--------------|--------------|---------------|------------|
| **Grad-ECLIP** | **0.856** | **0.782** | **0.734** | **0.689** | **15.2** |
| CLIP Surgery | 0.823 | 0.754 | 0.701 | 0.652 | 22.3 |
| GAME-MM | 0.798 | 0.721 | 0.678 | 0.634 | 18.7 |
| M2IB | 0.814 | 0.743 | 0.695 | 0.648 | 35.1 |
| BLIP | 0.789 | 0.712 | 0.665 | 0.621 | 28.9 |

### Visualisations de résultats

- **concept_decomposition.png** : Décomposition d'une image complexe en concepts visuels
- **map_comparaison.png** : Comparaison visuelle côte-à-côte des 5 méthodes
- **textual_explanation.png** : Explications de la modalité textuelle

Les résultats détaillés sont sauvegardés dans `insertion_evaluation_results.csv`.

## 📊 Documentation

### Documents principaux

#### rapport_projet_bgdia708_grad_clip.pdf
**Rapport complet du projet** incluant :
- Introduction et motivation  
- État de l'art des méthodes d'explicabilité
- Méthodologie et implémentation
- Expérimentations et évaluation
- Résultats et analyse comparative
- Discussion et perspectives

#### finetuning.md
**Guide de fine-tuning** pour adapter les modèles à des domaines spécifiques.

#### 2502.18816v1.pdf
**Article scientifique de référence** décrivant la méthode théorique Grad-ECLIP.

### Scripts utilitaires

- **`generate_emap.py`** : Implémentation principale de l'algorithme Grad-ECLIP
- **`clip_utils.py`** : Fonctions utilitaires pour CLIP
- **`imagenet_metadata.py`** : Gestion des métadonnées ImageNet
- **`test.py`** : Scripts de test et validation

## 🔬 Développement

### Notebooks de développement

- **`adaptation_vit.ipynb`** : Expérimentations sur l'adaptation des Vision Transformers
- **`finetuning.ipynb`** : Fine-tuning des modèles pour des domaines spécifiques

### Checkpoints et sauvegarde

Le dossier `pynvml_checkpoints/` contient les points de contrôle d'entraînement et modèles fine-tunés.

## 📚 Références

### Citation de l'article original

```bibtex
@article{zhao2024gradient,
  title={Gradient-based Visual Explanation for CLIP},
  author={Zhao, Chenyang and Wang, Kun and others},
  journal={arXiv preprint arXiv:2502.18816},
  year={2024}
}
```

### Repository officiel

Ce projet s'inspire du repository officiel de Grad-ECLIP :
- **URL** : https://github.com/Cyang-Zhao/Grad-Eclip
- **Auteurs** : Cyang-Zhao et collaborateurs
- **Licence** : Selon les termes du repository original

### Datasets

- **ImageNet ILSVRC2012** : http://www.image-net.org/
- **MS-COCO** : https://cocodataset.org/
- **Conceptual Captions** : https://ai.google.com/research/ConceptualCaptions/

---

## 🎯 Objectif du projet

Ce projet académique vise à :
1. **Reproduire** fidèlement la méthode Grad-ECLIP
2. **Comparer** avec les méthodes d'explicabilité existantes
3. **Évaluer** quantitativement sur ImageNet
4. **Documenter** complètement l'implémentation

Le projet fournit une base solide pour comprendre et étendre les méthodes d'explicabilité pour les modèles vision-langage.

**📊 Status** : Implémentation complète | Évaluation terminée | Documentation finalisée