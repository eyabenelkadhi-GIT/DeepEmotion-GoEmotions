# 🔄 Workflow : Des Notebooks à l'Interface Streamlit

## 📋 Vue d'ensemble

```
ÉTAPE 1: Préparation       → Notebook_0 (Personne 1)
ÉTAPE 2: Entraînement      → Notebooks 1-4 (3 personnes en parallèle)
ÉTAPE 3: Comparaison       → Notebook_5 (Tous ensemble)
ÉTAPE 4: Interface         → Streamlit (Sur PC local)
```

---

## 🎯 ÉTAPE 1 : Préparation des Données (Personne 1)

### Qui fait quoi ?
**UN SEUL MEMBRE** (Personne 1) exécute **Personne_1_Preparation_Donnees.ipynb**

### Pourquoi ?
- ✅ Gagner du temps (pas besoin de télécharger 3 fois le dataset)
- ✅ Assurer la cohérence (même tokenizer pour tous)
- ✅ Éviter les duplications

### Actions de Personne 1 :
1. Ouvrir **Personne_1_Preparation_Donnees.ipynb** sur Google Colab
2. **Télécharger dataset depuis Kaggle** :
   - Aller sur https://www.kaggle.com/datasets/debarshichanda/goemotions/data
   - Cliquer sur "Download" → télécharger `goemotions.csv`
   - Uploader dans Colab (via le bouton upload dans la première cellule)
3. Exécuter toutes les cellules
4. Télécharger les fichiers générés :
   ```python
   # Dans la dernière cellule de Notebook_0
   from google.colab import files
   !zip -r data_processed.zip /content/data/processed/
   files.download('data_processed.zip')
   ```
5. Partager `data_processed.zip` sur Google Drive avec l'équipe

### Actions de Personne 2 et 3 :
1. Télécharger `data_processed.zip` depuis Drive
2. Dans chaque notebook (1-4), ajouter cette cellule au début :
   ```python
   # Uploader le fichier data_processed.zip
   from google.colab import files
   uploaded = files.upload()  # Sélectionner data_processed.zip
   
   # Extraire
   !unzip -q data_processed.zip -d /content/
   
   # Vérifier
   !ls /content/data/processed/
   ```

---

## 🚀 ÉTAPE 2 : Entraînement des Modèles

### Chaque membre exécute ses notebooks

**Personne 1 :**
- Personne_1_LSTM.ipynb

**Personne 2 :**
- Personne_2_BiLSTM_Attention.ipynb
- Personne_2_CNN_BiLSTM_Attention.ipynb

**Personne 3 :**
- Personne_3_BERT.ipynb (avec GPU)

### Important : Sauvegarder les modèles
Chaque notebook sauvegarde automatiquement dans `/content/models/{nom_modele}/` :
- `model.h5` : Le modèle entraîné
- `results.json` : Métriques (F1, precision, recall, etc.)
- `predictions.npy` : Prédictions sur test set

### Télécharger les modèles après entraînement
À la fin de chaque notebook, ajouter :
```python
# Télécharger le modèle entraîné
from google.colab import files

# Pour LSTM (Personne 1)
!zip -r lstm_model.zip /content/models/lstm/
files.download('lstm_model.zip')

# Pour BiLSTM (Personne 2)
!zip -r bilstm_model.zip /content/models/bilstm/
files.download('bilstm_model.zip')

# Pour CNN-BiLSTM (Personne 2)
!zip -r cnn_bilstm_model.zip /content/models/cnn_bilstm/
files.download('cnn_bilstm_model.zip')

# Pour BERT (Personne 3)
!zip -r bert_model.zip /content/models/bert/
files.download('bert_model.zip')
```

---

## 📊 ÉTAPE 3 : Comparaison Finale (Tous ensemble)

### Préparation
1. Chaque membre partage son fichier .zip sur Drive
2. Un membre rassemble tous les modèles dans un dossier Drive commun

### Dans Tous_Comparaison_Finale.ipynb
1. Uploader tous les fichiers .zip des modèles
2. Extraire :
   ```python
   !unzip -q lstm_model.zip -d /content/
   !unzip -q bilstm_model.zip -d /content/
   !unzip -q cnn_bilstm_model.zip -d /content/
   !unzip -q bert_model.zip -d /content/
   ```
3. Exécuter Notebook_5 pour la comparaison

---

## 🎨 ÉTAPE 4 : Interface Streamlit (Sur PC Local)

### Préparation de l'environnement

1. **Créer la structure locale**
   ```
   projet/
   ├── app/
   │   └── streamlit_app.py    (déjà créé ✅)
   ├── models/
   │   ├── lstm/
   │   │   ├── model.h5
   │   │   └── tokenizer.pkl
   │   ├── bilstm/
   │   │   └── model.h5
   │   ├── cnn_bilstm/
   │   │   └── model.h5
   │   └── bert/
   │       └── model.h5
   └── data/
       └── processed/
           └── tokenizer.pkl
   ```

2. **Télécharger et extraire tous les modèles**
   - Télécharger les 4 fichiers .zip depuis Colab
   - Extraire dans le dossier `projet/models/`

3. **Copier le tokenizer**
   ```bash
   # Le tokenizer est dans data_processed.zip
   # Copier tokenizer.pkl dans models/lstm/ et data/processed/
   ```

### Lancement de l'interface

1. **Installation des dépendances**
   ```bash
   cd projet/
   pip install -r requirements.txt
   ```

2. **Vérifier que les modèles sont présents**
   ```bash
   # Windows PowerShell
   Get-ChildItem models -Recurse -Filter *.h5
   
   # Devrait afficher :
   # models/lstm/model.h5
   # models/bilstm/model.h5
   # models/cnn_bilstm/model.h5
   # models/bert/model.h5
   ```

3. **Lancer Streamlit**
   ```bash
   cd projet/
   streamlit run app/streamlit_app.py
   ```

4. **Accéder à l'interface**
   - Ouvrir automatiquement : http://localhost:8501
   - Ou manuellement dans le navigateur

### Utilisation de l'interface

1. **Sélectionner un modèle** (sidebar gauche)
   - LSTM
   - BiLSTM + Attention
   - CNN-BiLSTM + Attention
   - BERT

2. **Entrer un texte ou choisir un exemple**
   - Ex: "I'm so happy and excited about this amazing news!"

3. **Cliquer sur "Analyser"**
   - L'interface charge le modèle
   - Prédit les émotions
   - Affiche les résultats

4. **Explorer les visualisations**
   - Top 10 émotions (barres horizontales)
   - Distribution complète (28 émotions)
   - Radar chart

5. **Prendre des captures d'écran** pour le rapport

---

## 📸 Captures d'Écran pour le Rapport

### À capturer :
1. **Page principale** avec sélection de modèle
2. **Exemple de prédiction** (texte joyeux → émotion "joy")
3. **Top 10 émotions** (graphique en barres)
4. **Radar chart**
5. **Distribution des 28 émotions**
6. **Comparaison** entre 2 modèles (même texte, modèles différents)

---

## 🔧 Dépannage

### Problème : Modèle introuvable
```
❌ Erreur : Impossible de charger le modèle LSTM
```
**Solution :**
- Vérifier que `models/lstm/model.h5` existe
- Vérifier le chemin dans `streamlit_app.py` (ligne 76-81)

### Problème : Tokenizer introuvable
```
⚠️ Tokenizer non trouvé. Créer un tokenizer de base.
```
**Solution :**
- Copier `tokenizer.pkl` depuis `data_processed.zip`
- Placer dans `models/lstm/tokenizer.pkl` OU `data/processed/tokenizer.pkl`

### Problème : Importation TensorFlow
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution :**
```bash
pip install tensorflow==2.15.0
```

### Problème : Importation Plotly
```
ModuleNotFoundError: No module named 'plotly'
```
**Solution :**
```bash
pip install plotly streamlit
```

---

## ✅ Checklist Complète

### Phase 1 : Données
- [ ] Personne 1 exécute Notebook_0
- [ ] Personne 1 télécharge data_processed.zip
- [ ] Personne 1 partage sur Drive
- [ ] Personnes 2 et 3 téléchargent data_processed.zip

### Phase 2 : Entraînement
- [ ] Personne 1 entraîne LSTM → télécharge lstm_model.zip
- [ ] Personne 2 entraîne BiLSTM → télécharge bilstm_model.zip
- [ ] Personne 2 entraîne CNN-BiLSTM → télécharge cnn_bilstm_model.zip
- [ ] Personne 3 entraîne BERT → télécharge bert_model.zip
- [ ] Tous partagent leurs .zip sur Drive

### Phase 3 : Comparaison
- [ ] Rassembler tous les modèles
- [ ] Exécuter Notebook_5
- [ ] Analyser les résultats

### Phase 4 : Interface
- [ ] Extraire tous les .zip dans projet/models/
- [ ] Copier tokenizer.pkl dans les bons dossiers
- [ ] Installer dépendances : `pip install -r requirements.txt`
- [ ] Lancer : `streamlit run app/streamlit_app.py`
- [ ] Tester avec les 4 modèles
- [ ] Prendre 6+ captures d'écran
- [ ] Inclure dans le rapport

---

## 🎯 Résumé

| Étape | Qui | Où | Durée |
|-------|-----|-----|-------|
| 1. Données | Personne 1 | Colab | 15-20 min |
| 2. LSTM | Personne 1 | Colab | 15-20 min |
| 2. BiLSTM + CNN | Personne 2 | Colab | 1h10 |
| 2. BERT | Personne 3 | Colab | 1h |
| 3. Comparaison | Tous | Colab | 15 min |
| 4. Interface | Tous | PC local | 30 min setup + démo |

**Temps total : ~3h sur Colab + 30 min interface**

---

## 💡 Conseils

1. **Communication** : Créer un groupe WhatsApp/Discord pour coordination
2. **Partage** : Utiliser Google Drive partagé dès le début
3. **Sauvegarde** : Télécharger TOUJOURS les modèles après entraînement
4. **Test** : Tester l'interface AVANT la présentation finale
5. **Backup** : Garder une copie de tous les .zip sur Drive
