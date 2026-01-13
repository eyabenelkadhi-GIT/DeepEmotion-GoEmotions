# 📋 Guide Rapide : Quel Notebook pour Quelle Personne ?

## 🎯 Répartition Claire des Notebooks

### 👤 PERSONNE 1 (Responsable Données + LSTM)

#### 📂 Ses notebooks :
1. **Personne_1_Preparation_Donnees.ipynb** ⭐ PRIORITÉ ABSOLUE
   - ⚠️ **À FAIRE EN PREMIER**
   - UN SEUL membre fait ce notebook
   - Télécharge dataset depuis Kaggle
   - Prépare les données pour tout le monde
   - Partage `data_processed.zip` sur Drive
   - ⏱️ Temps : 15-20 minutes

2. **Personne_1_LSTM.ipynb**
   - Modèle baseline (LSTM simple)
   - Après avoir fini la préparation des données
   - ⏱️ Temps : 15-20 minutes

**Total Personne 1 : ~40 minutes**

---

### 👤 PERSONNE 2 (Modèles Hybrides)

#### 📂 Ses notebooks :
1. **Personne_2_BiLSTM_Attention.ipynb**
   - BiLSTM avec mécanisme d'attention
   - ⚠️ Attendre que Personne 1 partage `data_processed.zip`
   - ⏱️ Temps : 30-40 minutes

2. **Personne_2_CNN_BiLSTM_Attention.ipynb**
   - Architecture hybride CNN + BiLSTM + Attention
   - Après BiLSTM
   - ⏱️ Temps : 40-50 minutes

**Total Personne 2 : ~1h10-1h30**

---

### 👤 PERSONNE 3 (Transfer Learning)

#### 📂 Son notebook :
1. **Personne_3_BERT.ipynb**
   - Fine-tuning BERT-base-uncased
   - ⚠️ Attendre que Personne 1 partage `data_processed.zip`
   - ⚠️ **NÉCESSITE GPU** (activer sur Colab)
   - ⏱️ Temps : 50-60 minutes

**Total Personne 3 : ~1h**

---

### 👥 TOUS ENSEMBLE (Comparaison)

#### 📂 Notebook final :
1. **Tous_Comparaison_Finale.ipynb**
   - ⚠️ Après que les 3 membres aient fini leurs modèles
   - Compare les 4 modèles
   - Étude d'ablation
   - Génère le rapport de comparaison
   - ⏱️ Temps : 10-15 minutes

---

## 🔄 Ordre d'Exécution

```
JOUR 1 - MATIN
└── Personne 1 : Personne_1_Preparation_Donnees.ipynb (15-20 min)
    └── Partage data_processed.zip sur Drive
    └── Notifie l'équipe ✅

JOUR 1 - APRÈS-MIDI (EN PARALLÈLE)
├── Personne 1 : Personne_1_LSTM.ipynb (15-20 min)
├── Personne 2 : Personne_2_BiLSTM_Attention.ipynb (30-40 min)
└── Personne 3 : Personne_3_BERT.ipynb (50-60 min)

JOUR 2 - MATIN
└── Personne 2 : Personne_2_CNN_BiLSTM_Attention.ipynb (40-50 min)

JOUR 2 - APRÈS-MIDI
└── Tous : Tous_Comparaison_Finale.ipynb (10-15 min)
```

---

## 📊 Résumé Visuel

| Personne | Notebooks | Temps Total | Ordre |
|----------|-----------|-------------|-------|
| **Personne 1** | Personne_1_Preparation_Donnees + Personne_1_LSTM | ~40 min | 1er (préparation) puis en parallèle |
| **Personne 2** | Personne_2_BiLSTM + Personne_2_CNN_BiLSTM | ~1h30 | En parallèle puis séquentiel |
| **Personne 3** | Personne_3_BERT | ~1h | En parallèle |
| **Tous** | Tous_Comparaison_Finale | ~15 min | Dernier |

---

## ✅ Checklist par Personne

### Personne 1 ✅
- [ ] Télécharger dataset depuis Kaggle
- [ ] Exécuter `Personne_1_Preparation_Donnees.ipynb`
- [ ] Télécharger `data_processed.zip` depuis Colab
- [ ] Partager sur Drive + envoyer lien à l'équipe
- [ ] Exécuter `Personne_1_LSTM.ipynb`
- [ ] Télécharger `lstm_model.zip`
- [ ] Partager sur Drive

### Personne 2 ✅
- [ ] Attendre notification de Personne 1
- [ ] Télécharger `data_processed.zip` depuis Drive
- [ ] Uploader dans Colab
- [ ] Exécuter `Personne_2_BiLSTM_Attention.ipynb`
- [ ] Télécharger `bilstm_model.zip`
- [ ] Exécuter `Personne_2_CNN_BiLSTM_Attention.ipynb`
- [ ] Télécharger `cnn_bilstm_model.zip`
- [ ] Partager les 2 fichiers sur Drive

### Personne 3 ✅
- [ ] Attendre notification de Personne 1
- [ ] Télécharger `data_processed.zip` depuis Drive
- [ ] Uploader dans Colab
- [ ] Activer GPU sur Colab (Runtime → Change runtime type → GPU)
- [ ] Exécuter `Personne_3_BERT.ipynb`
- [ ] Télécharger `bert_model.zip`
- [ ] Partager sur Drive

### Tous Ensemble ✅
- [ ] Vérifier que les 4 modèles sont sur Drive
- [ ] Télécharger tous les .zip
- [ ] Uploader dans Colab
- [ ] Exécuter `Tous_Comparaison_Finale.ipynb`
- [ ] Analyser les résultats
- [ ] Prendre des captures d'écran

---

## 🚨 Points d'Attention

### ⚠️ CRITIQUE
1. **Personne 1 DOIT finir en premier** - Les autres attendent
2. **UN SEUL membre** prépare les données (pas de duplication)
3. **Personne 3 DOIT activer GPU** sinon BERT sera très lent

### 💡 CONSEILS
1. **Communication** : Créer un groupe WhatsApp/Discord
2. **Notifications** : Prévenir quand chaque étape est finie
3. **Drive** : Créer un dossier partagé dès le début
4. **Sauvegarde** : Toujours télécharger les modèles après entraînement

---

## 📱 Messages à Envoyer dans le Groupe

### Message 1 (Personne 1)
```
✅ J'ai fini Personne_1_Preparation_Donnees.ipynb !
📦 data_processed.zip uploadé sur Drive
🔗 Lien : [insérer lien Drive]
👉 Vous pouvez commencer vos notebooks !
```

### Message 2 (Chaque personne après son modèle)
```
✅ [Personne 1/2/3] : Mon modèle est entraîné !
📦 [lstm/bilstm/cnn_bilstm/bert]_model.zip sur Drive
🔗 Lien : [insérer lien Drive]
```

### Message 3 (Quand tous ont fini)
```
🎉 Les 4 modèles sont prêts !
👥 On se retrouve pour Tous_Comparaison_Finale.ipynb ?
```

---

## 🎯 Structure Finale des Fichiers

```
Google Drive Partagé/
├── data_processed.zip           (Personne 1)
├── lstm_model.zip               (Personne 1)
├── bilstm_model.zip             (Personne 2)
├── cnn_bilstm_model.zip         (Personne 2)
└── bert_model.zip               (Personne 3)
```

---

## 💻 Noms des Notebooks dans Colab

Quand vous uploadez les notebooks sur Google Colab, vous verrez :

```
Mes Notebooks/
├── Personne_1_Preparation_Donnees.ipynb    ← Personne 1 uniquement
├── Personne_1_LSTM.ipynb                   ← Personne 1 uniquement
├── Personne_2_BiLSTM_Attention.ipynb       ← Personne 2 uniquement
├── Personne_2_CNN_BiLSTM_Attention.ipynb   ← Personne 2 uniquement
├── Personne_3_BERT.ipynb                   ← Personne 3 uniquement
└── Tous_Comparaison_Finale.ipynb           ← Tous ensemble
```

**Chaque personne voit clairement ses notebooks grâce au préfixe !**

---

## 🎓 Résumé Ultra-Simple

| Si tu es... | Tu fais... | Dans cet ordre... |
|-------------|------------|-------------------|
| **Personne 1** | 2 notebooks | Préparation (d'abord) → LSTM (après) |
| **Personne 2** | 2 notebooks | BiLSTM (d'abord) → CNN-BiLSTM (après) |
| **Personne 3** | 1 notebook | BERT (avec GPU activé) |
| **Tous** | 1 notebook | Comparaison (à la fin) |

**Total : 6 notebooks, ~3 heures de travail, répartis sur 2-3 jours** 🚀
