# 🚀 Guide de Travail en Équipe - Projet Détection d'Émotions

## 📋 Organisation pour 3 Membres

### **Personne 1 : Préparation + Modèle Simple** ⭐ RESPONSABLE DONNÉES
- **Personne_1_Preparation_Donnees.ipynb** (⚠️ PRIORITÉ - À faire en premier)
  - Téléchargement du dataset GoEmotions depuis Kaggle (58k commentaires)
  - Source : https://www.kaggle.com/datasets/debarshichanda/goemotions/data
  - Analyse exploratoire des données (EDA)
  - Prétraitement et tokenization
  - Sauvegarde des données pour toute l'équipe
  - ⏱️ Temps estimé : 15-20 minutes
  - ⚠️ **UN SEUL MEMBRE FAIT CETTE ÉTAPE** - Les autres attendent les fichiers
  
- **Personne_1_LSTM.ipynb** (Modèle baseline)
  - Entraînement LSTM simple (64 unités)
  - Évaluation et métriques
  - ⏱️ Temps estimé : 15-20 minutes

### **Personne 2 : Modèles Hybrides**
- **Personne_2_BiLSTM_Attention.ipynb**
  - BiLSTM avec mécanisme d'attention custom
  - ⏱️ Temps estimé : 30-40 minutes
  
- **Personne_2_CNN_BiLSTM_Attention.ipynb**
  - Architecture hybride CNN + BiLSTM + Attention
  - ⏱️ Temps estimé : 40-50 minutes

### **Personne 3 : Transfer Learning**
- **Personne_3_BERT.ipynb**
  - Fine-tuning BERT (bert-base-uncased)
  - ⚠️ **Nécessite GPU obligatoire**
  - ⏱️ Temps estimé : 50-60 minutes

### **Tous ensemble : Comparaison Finale**
- **Tous_Comparaison_Finale.ipynb**
  - Chargement de tous les résultats
  - Comparaison des 4 modèles
  - Étude d'ablation
  - Génération du rapport final
  - ⏱️ Temps estimé : 10-15 minutes

---

## 🔄 Workflow Parallèle

### Phase 1 : Setup Initial (Personne 1 UNIQUEMENT) 🔴
```
Personne 1 : 
1. Télécharge dataset depuis Kaggle
2. Exécute Notebook_0 
3. Partage le dossier data/processed/ sur Google Drive
```

**⚠️ IMPORTANT** : 
- **UN SEUL membre** (Personne 1) prépare les données
- Les Personnes 2 et 3 **NE FONT PAS** Notebook_0
- Tous utilisent les MÊMES données préparées par Personne 1
- **Raison** : Gagner du temps, assurer la cohérence des données

### Phase 2 : Entraînement Parallèle (Après préparation des données)
```
Personne 1 : Personne_1_LSTM.ipynb
Personne 2 : Personne_2_BiLSTM_Attention.ipynb → Personne_2_CNN_BiLSTM_Attention.ipynb  (en séquence)
Personne 3 : Personne_3_BERT.ipynb
```

**Partage requis :**
- Personne 1 partage : `data/processed/` (tokenizer.pkl, X_train.npy, etc.)
- Chacun partage après : `models/{lstm,bilstm,cnn_bilstm,bert}/results.json`

### Phase 3 : Comparaison Finale (Tous ensemble)
```
Tous : Tous_Comparaison_Finale.ipynb
```

---

## 📦 Partage des Fichiers (Google Drive/Colab)

### Fichiers à partager par Personne 1 (OBLIGATOIRE)
```
data/processed/
├── X_train.npy
├── X_val.npy
├── X_test.npy
├── y_train.npy
├── y_val.npy
├── y_test.npy
├── tokenizer.pkl
└── metadata.pkl
```

### Fichiers à partager par chaque personne après entraînement
```
Personne 1:
models/lstm/
├── model.h5
├── results.json
└── predictions.npy

Personne 2:
models/bilstm/
models/cnn_bilstm/

Personne 3:
models/bert/
```

---

## ⚙️ Configuration Google Colab

### Pour TOUTES les personnes :

1. **Activer le GPU** (surtout Personne 3 pour BERT)
   ```
   Runtime → Change runtime type → GPU (T4)
   ```

2. **Monter Google Drive** (dans chaque notebook)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Copier les données partagées**
   ```python
   # Après que Personne 1 ait fini Notebook_0
   !cp -r /content/drive/MyDrive/projet_emotions/data/processed /content/data/
   ```

---

## 🎯 Checklist d'Exécution

### Personne 1 ✅
- [ ] Exécuter Personne_1_Preparation_Donnees.ipynb (préparer données)
- [ ] Uploader data/processed/ sur Drive
- [ ] Partager le lien avec l'équipe
- [ ] Exécuter Personne_1_LSTM.ipynb (LSTM)
- [ ] Uploader models/lstm/ sur Drive

### Personne 2 ✅
- [ ] Attendre data/processed/ de Personne 1
- [ ] Télécharger data/processed/
- [ ] Exécuter Personne_2_BiLSTM_Attention.ipynb
- [ ] Exécuter Personne_2_CNN_BiLSTM_Attention.ipynb
- [ ] Uploader models/bilstm/ et models/cnn_bilstm/

### Personne 3 ✅
- [ ] Attendre data/processed/ de Personne 1
- [ ] Activer GPU sur Colab
- [ ] Télécharger data/processed/
- [ ] Exécuter Personne_3_BERT.ipynb
- [ ] Uploader models/bert/

### Tous Ensemble ✅
- [ ] Rassembler tous les résultats (4 modèles)
- [ ] Exécuter Tous_Comparaison_Finale.ipynb
- [ ] Analyser les résultats
- [ ] Préparer le rapport final

---

## 📊 Interface Streamlit (Bonus +10-15%) 🎨

### Comment ça fonctionne ?
L'interface Streamlit charge automatiquement les modèles entraînés par vos notebooks :

```
Notebook_1 → Sauvegarde models/lstm/model.h5
Notebook_2 → Sauvegarde models/bilstm/model.h5
Notebook_3 → Sauvegarde models/cnn_bilstm/model.h5
Notebook_4 → Sauvegarde models/bert/model.h5
                    ↓
Interface Streamlit charge ces fichiers .h5
                    ↓
L'utilisateur entre un texte → Prédiction en temps réel
```

### Étapes pour créer l'interface

**Après avoir entraîné tous les modèles :**

1. **Télécharger les modèles depuis Colab vers votre PC**
   ```python
   # Dans Colab, après chaque notebook
   from google.colab import files
   !zip -r models.zip /content/models/
   files.download('models.zip')
   ```

2. **Extraire les modèles dans le projet**
   ```
   projet/
   └── models/
       ├── lstm/
       │   ├── model.h5
       │   └── tokenizer.pkl
       ├── bilstm/
       │   └── model.h5
       ├── cnn_bilstm/
       │   └── model.h5
       └── bert/
           └── model.h5
   ```

3. **Installer Streamlit**
   ```bash
   pip install streamlit plotly
   ```

4. **Lancer l'interface**
   ```bash
   cd projet/
   streamlit run app/streamlit_app.py
   ```

5. **Accéder à l'interface**
   - Ouvrir : http://localhost:8501
   - Tester avec des exemples
   - Prendre des captures d'écran pour le rapport

### Fonctionnalités de l'interface ✅
- ✅ Sélection des 4 modèles
- ✅ Zone de texte pour entrée utilisateur
- ✅ Prédiction en temps réel
- ✅ Top 10 émotions avec barres horizontales
- ✅ Radar chart des émotions
- ✅ Distribution complète des 28 émotions
- ✅ Exemples prédéfinis
- ✅ Tableau détaillé des probabilités

**L'interface est DÉJÀ codée dans `app/streamlit_app.py` !**

---

## 🚨 Résolution de Problèmes

### Problème : Notebook ne trouve pas les données
```python
# Vérifier que data/processed/ existe
import os
print(os.listdir('/content/data/processed/'))
```

### Problème : BERT trop lent
- Vérifier que GPU est activé : `Runtime → Change runtime type → GPU`
- Réduire batch_size de 32 à 16

### Problème : Out of Memory
- Redémarrer le runtime : `Runtime → Factory reset runtime`
- Réexécuter depuis le début

---

## ⏱️ Timeline Recommandé

| Jour | Personne 1 | Personne 2 | Personne 3 | Équipe |
|------|------------|------------|------------|--------|
| J1 | Notebook_0 + partage | Attente | Attente | - |
| J2 | Notebook_1 | Notebook_2 | Notebook_4 | - |
| J3 | Rapport (intro) | Notebook_3 | Test BERT | - |
| J4 | - | Rapport (méthode) | Rapport (résultats) | Notebook_5 |
| J5 | - | - | - | Interface Streamlit |

**Temps total estimé : 2h30-3h d'entraînement + rédaction rapport**

---

## 📞 Communication

**Communication essentielle :**
1. Personne 1 notifie quand Notebook_0 est terminé ✅
2. Chacun notifie quand son modèle est entraîné ✅
3. Rassemblement pour Notebook_5 quand tous les modèles sont prêts ✅

**Outils recommandés :** WhatsApp/Discord pour coordination temps réel
