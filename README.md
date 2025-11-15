# Grad-ECLIP: Explainable CLIP via Gradient-based Attention Analysis

Ce projet reproduit et implémente la méthode **Grad-ECLIP** décrite dans l'article scientifique 2502.18816v1.pdf, qui propose une approche novatrice pour expliquer les décisions du modèle CLIP (Contrastive Language-Image Pre-Training) en utilisant les gradients des couches d'attention.

## Liste des auteurs par ordre de contribution au projet
Les contributeurs sont présentés ci-dessous par ordre décroissant de leur niveau d'implication dans le projet :

* Mbathe Mekontchou Paul (Contributeur principal)
* Ouhiba Aymen
* Wande Wula Alfred
* Vu Julien
* Garra Nohalia


##  Table des matières

- Vue d'ensemble
- Structure du projet
- Installation
- Téléchargement des données
- Notebooks principaux
- Méthodes d'explication comparées
- Évaluation sur ImageNet
- Documentation et rapport
- Contribution
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
-  **Interprétabilité** : Visualisations claires et intuitives

##  Structure du projet

```
├── 2502.18816v1.pdf                    #  Article scientifique de référence
├── rapport_projet_bgdia708_grad_clip.pdf  #  Rapport complet du projet
├── README.md                           #  Ce fichier
├── requirements.txt                    #  Dépendances Python
├── download_dataset.ipynb              # Notebook pour le téléchargement des datasets
├── valprep.sh                         #  Script d'organisation ImageNet
├── imagenet_class_index.json          #  Index des classes ImageNet
├── imagenet_labels.txt                #  Labels ImageNet
├── textual_explanation.png            #  Explications textuelles
├── whippet.png                        #  Image d'exemple
│
├── CLIP/                              # Implementation CLIP originale
│   ├── clip/                          # Module CLIP core
│   ├── notebooks/                     # Notebooks d'exemple CLIP
│   └── requirements.txt               # Dépendances CLIP
│
├── Grad_ECLIP/                         # 🚀 Notre implementation principale
│   ├──  Notebooks d'explication
│   │   ├── grad_eclip_image.ipynb     # Explication image→texte
│   │   ├── grad_eclip_text.ipynb      # Explication texte→image
│   │   └── compare_visualize.ipynb     # Comparaison des méthodes
│   │
│   ├──  Notebooks d'évaluation
│   │   ├── imagenet_eval_deletion.ipynb    # Test de suppression
│   │   ├── imagenet_eval_insertion.ipynb   # Test d'insertion
│   │   └── grad_eclip_finetuning.ipynb               # Fine-tuning des modèles
│   │
│   ├──  Scripts utilitaires
│   │   ├── clip_utils.py               # Utilitaires CLIP
│   │   ├── generate_emap.py            # Génération des cartes d'explication
│   │   ├── imagenet_metadata.py        # Métadonnées ImageNet
│   │   ├── insertion_evaluation_results.csv # Résultats évaluation
│   │   └── test.py                     # Scripts de test
│   │
│   ├──  Méthodes comparées
│   │   ├── BLIP/                       # BLIP implementation
│   │   │   ├── blip_vit.py
│   │   │   ├── med.py
│   │   │   └── vit.py
│   │   ├── CLIP_Surgery/               # CLIP Surgery method
│   │   │   ├── clip_utils.py
│   │   │   └── pytorch_clip_guided_diffusion/
│   │   ├── Game_MM_CLIP/              # GAME-MM method
│   │   │   ├── models/
│   │   │   └── utils/
│   │   └── M2IB/                      # M2IB method
│   │       ├── model.py
│   │       └── utils.py
│   │
│   ├──  Données et résultats
│   │   ├── data/val/                   # Dataset de validation ImageNet
│   │   ├── images/                     # Images d'exemple
│   │   └── outfile/                    # Résultats de sortie
│   │
│   └──  Notebooks de développement
│       ├── pynvml_checkpoints/         # Points de contrôle
│       └── adaptation_vit.ipynb        # Adaptation Vision Transformer
│
└── outfile/                           #  Résultats globaux du projet
```

## Installation

### Prérequis

- **Python 3.8+**
- **CUDA 11.0+** (recommandé pour GPU)
- **Git**
- **Jupyter Notebook**
- **8GB+ RAM** (16GB recommandé)
- **GPU avec 4GB+ VRAM** (optionnel mais recommandé)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/mbathe/Projet-IA-Fairness.git
cd Projet-IA-Fairness

# Créer un environnement virtuel (recommandé)
python -m venv grad_eclip_env
source grad_eclip_env/bin/activate  # Linux/Mac
# ou
grad_eclip_env\Scripts\activate     # Windows

# Installer les dépendances principales
pip install -r requirements.txt

# Installer CLIP
cd CLIP
pip install -r requirements.txt
cd ..

# Installer les dépendances spécifiques Grad-ECLIP
cd Grad_CLIP
pip install torch torchvision torchaudio
pip install transformers
pip install opencv-python
pip install matplotlib seaborn
pip install numpy pandas
pip install scikit-learn
pip install Pillow
pip install tqdm
cd ..
```

### Vérification de l'installation

```bash
# Test rapide
python -c "import torch; import clip; print('Installation réussie!')"
```

##  Téléchargement des données

### Dataset principal : ImageNet

Le projet utilise **ImageNet ILSVRC2012** pour l'évaluation quantitative et le dataset **MS COCO** pour l'évaluation de l'approche de finetuning. Utilisez le notebook download_dataset.ipynb pour télécharger automatiquement les données :


### Datasets supplémentaires

Le projet supporte également :
- **MS-COCO** : Pour l'évaluation sur des scènes complexes
- **ImageNet-V2** : Version améliorée d'ImageNet
- **Conceptual Captions** : Paires image-texte

## Notebooks principaux

### 1. `grad_eclip_image.ipynb` 
** Explication des images par le texte**

Ce notebook implémente l'algorithme principal de Grad-ECLIP pour expliquer pourquoi une image correspond à un texte donné.

**Fonctionnalités :**
- Génération de cartes de saillance pour les images
- Visualisation des régions importantes avec heatmaps
- Superposition des explications sur l'image originale
- Comparaison avec les méthodes baseline
- Export des résultats en haute résolution



**Sorties générées :**
- Cartes de saillance colorées
- Images avec régions importantes surlignées  
- Graphiques de distribution des scores d'attention
- Comparaisons côte-à-côte avec autres méthodes

### 2. `grad_eclip_text.ipynb`
**Explication du texte par l'image**

Ce notebook implémente l'explication inverse : quels mots du texte sont importants pour la correspondance avec l'image.

**Fonctionnalités :**
- Attribution d'importance aux tokens textuels
- Visualisation des mots-clés avec codes couleur
- Analyse de la contribution de chaque mot
- Génération de nuages de mots pondérés
- Export des explications textuelles

**Exemple d'utilisation :**
```python
# Expliquer l'importance des mots
text = "a small brown and white dog running in the grass"
image = load_image("dog_running.jpg")

# Générer l'explication textuelle
text_explanation = grad_eclip_explain_text(model, image, text)

# Visualiser les mots importants
visualize_text_importance(text, text_explanation)
# Sortie : "a small [BROWN] and [WHITE] [DOG] running in the [GRASS]"
# (mots en majuscules = plus importants)
```

**Analyses disponibles :**
- Heatmap des tokens par importance
- Graphiques de contribution relative
- Analyse syntaxique des mots importants
- Comparaison avec les modèles de langue

### 3. `compare_visualize.ipynb`
**🔍 Comparaison des méthodes d'explication**

Ce notebook compare Grad-ECLIP avec les autres méthodes d'explicabilité disponibles sur les mêmes exemples.

**Méthodes comparées :**
- **Grad-ECLIP** (notre méthode)
- **BLIP** : Bootstrapping Language-Image Pre-training
- **CLIP Surgery** : Modification architecturale de CLIP
- **GAME-MM** : Gradient-weighted Class Activation Mapping
- **M2IB** : Multi-Modal Information Bottleneck



##  Méthodes d'explication comparées

### Grad-ECLIP (Notre méthode) 🏆
**Localisation :** generate_emap.py

- **Principe** : Utilise les gradients des couches d'attention de CLIP
- **Innovation** : Première méthode à exploiter spécifiquement les gradients d'attention cross-modale
- **Avantages** : 
  - Simple à implémenter
  - Pas de modification du modèle original
  - Applicable aux deux modalités (image et texte)
  - Résultats interprétables
- **Algorithme** :
  ```python
  def grad_eclip(model, image, text):
      # 1. Forward pass avec gradient tracking
      logits = model(image, text)
      
      # 2. Backward pass pour calculer les gradients
      grad = torch.autograd.grad(logits, model.attention_weights)
      
      # 3. Pondération des cartes d'attention par les gradients
      explanation = grad * model.attention_weights
      
      return explanation
  ```

### BLIP 
**Localisation :** BLIP

- **Principe** : Bootstrap vision-language understanding avec captioning
- **Architecture** : Vision Transformer + BERT multimodal
- **Fichiers principaux** :
  - `blip_vit.py` : Vision Transformer adapté
  - `med.py` : Encodeur multimodal
  - `vit.py` : Implementation ViT

### CLIP Surgery 
**Localisation :** CLIP_Surgery

- **Principe** : Modification architecturale de CLIP pour améliorer la localisation
- **Méthode** : Remplace les couches d'attention par des versions "chirurgicales"
- **Fichiers** :
  - `clip_utils.py` : Utilitaires modifiés
  - `pytorch_clip_guided_diffusion/` : Integration avec diffusion

### GAME-MM 
**Localisation :** Game_MM_CLIP

- **Principe** : Gradient-weighted Class Activation Mapping pour le multimodal
- **Extension** : Adapte Grad-CAM aux modèles vision-langage
- **Structure** :
  - `models/` : Architectures de modèles
  - `utils/` : Fonctions utilitaires

### M2IB 
**Localisation :** M2IB

- **Principe** : Multi-Modal Information Bottleneck
- **Théorie** : Minimise l'information mutuelle tout en préservant la performance
- **Implémentation** :
  - `model.py` : Architecture M2IB
  - `utils.py` : Fonctions de support

##  Évaluation sur ImageNet

### Classes et templates ImageNet

Le projet utilise les **1000 classes standard d'ImageNet** avec **80 templates d'augmentation** optimisés pour CLIP :

```python
# Exemples de templates utilisés dans l'évaluation
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    # ... 80 templates au total pour robusesse
]

# Classes ImageNet (extrait)
classes = [
    'tench',           # n01440764
    'goldfish',        # n01443537  
    'great white shark', # n01484850
    # ... 1000 classes au total
]
```

### Tests de performance quantitative

#### 1. `imagenet_eval_deletion.ipynb`
** Test de suppression (Deletion Test)**

Ce test mesure la **baisse de performance** quand on supprime progressivement les régions les plus importantes identifiées par chaque méthode.



#### 2. `imagenet_eval_insertion.ipynb`
** Test d'insertion (Insertion Test)**

Ce test mesure l'**amélioration de performance** quand on révèle progressivement les régions importantes sur une image initialement masquée.






## 📊 Documentation et rapport

### Documents principaux

#### rapport_projet_bgdia708_grad_clip.pdf
**Rapport complet du projet** (8 pages) incluant :

1. **Introduction et motivation**
   - Contexte des modèles vision-langage
   - Problématique de l'explicabilité
   - Objectifs du projet

2. **État de l'art**
   - Méthodes d'explicabilité existantes
   - Limites des approches actuelles
   - Positionnement de Grad-ECLIP

3. **Méthodologie**
   - Description détaillée de l'algorithme
   - Choix de conception
   - Implementation technique

4. **Expérimentations**
   - Protocole d'évaluation
   - Datasets utilisés
   - Métriques de performance

5. **Résultats et analyse**
   - Comparaisons quantitatives
   - Études qualitatives
   - Études d'ablation

6. **Discussion et perspectives**
   - Limites identifiées
   - Améliorations possibles
   - Applications futures


#### 2502.18816v1.pdf
**Article scientifique de référence** :
- Theoretical foundations
- Algorithmic details
- Experimental validation


## 🔬 Recherche et développement

### Notebooks de développement

#### `adaptation_vit.ipynb`
Expérimentations sur l'adaptation des Vision Transformers :
- Modifications architecturales testées
- Impact sur la performance d'explication
- Optimisations computationnelles

#### `finetuning.ipynb`






## 📚 Références

### Citations principales

#### Article original
```bibtex
@article{zhao2024gradient,
  title={Gradient-based Visual Explanation for CLIP},
  author={Zhao, Chenyang and Wang, Kun and others},
  journal={arXiv preprint arXiv:2502.18816},
  year={2024}
}
```

#### Notre implémentation
```bibtex
@misc{pmbathe2024gradeclip,
  title={Grad-ECLIP: Implementation and Comparative Study},
  author={pmbathe-24},
  institution={INFRES},
  year={2024},
  url={https://github.com/username/Projet-IA-Fairness}
}
```

### Références connexes

1. **CLIP Original** : Radford et al., "Learning Transferable Visual Representations from Natural Language Supervision"
2. **Vision Transformers** : Dosovitskiy et al., "An Image is Worth 16x16 Words"
3. **Grad-CAM** : Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
4. **BLIP** : Li et al., "BLIP: Bootstrapping Language-Image Pre-training"
5. **Attention Visualization** : Abnar & Zuidema, "Quantifying Attention Flow"

### Datasets et benchmarks

- **ImageNet** : http://www.image-net.org/
- **MS-COCO** : https://cocodataset.org/
- **Conceptual Captions** : https://ai.google.com/research/ConceptualCaptions/

---



**🎯 Objectif** : Ce README fournit une documentation complète pour comprendre, utiliser et étendre le projet Grad-ECLIP. Pour toute question spécifique, consultez les notebooks
