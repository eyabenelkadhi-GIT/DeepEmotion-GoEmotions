# 📚 Projet Détection d'Émotions - GoEmotions

## 🎯 Vue d'ensemble
Projet de Deep Learning pour la détection de **28 émotions** dans du texte, basé sur le dataset **GoEmotions** (58,000 commentaires Reddit).

---

## 📂 Structure du Projet

```
projet/
├── notebooks/                          # 6 Notebooks Google Colab
│   ├── Notebook_0_Preparation_Donnees.ipynb
│   ├── Notebook_1_LSTM.ipynb
│   ├── Notebook_2_BiLSTM_Attention.ipynb
│   ├── Notebook_3_CNN_BiLSTM_Attention.ipynb
│   ├── Notebook_4_BERT.ipynb
│   ├── Notebook_5_Comparaison_Finale.ipynb
│   └── INTERFACE_STREAMLIT.md         # Guide de l'interface
│
├── app/
│   └── streamlit_app.py                # Interface interactive
│
├── README_EQUIPE.md                    # Guide de travail en équipe
├── requirements.txt                    # Dépendances Python
└── enonce.txt                          # Énoncé du projet
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Exécuter les Notebooks (Google Colab)
1. Uploader les 6 notebooks sur Google Colab
2. Activer le GPU : `Runtime → Change runtime type → GPU`
3. **UN SEUL MEMBRE** exécute **Notebook_0** (préparation des données)
4. Partager `data/processed/` avec les autres membres
5. Exécuter Notebooks 1-4 (entraînement des modèles en parallèle)
6. Exécuter **Notebook_5** (comparaison finale)

**📖 Guide complet** : Voir [WORKFLOW_COMPLET.md](WORKFLOW_COMPLET.md)

### 3. Lancer l'Interface Streamlit (après entraînement)
1. Télécharger les modèles depuis Colab vers PC
2. Placer dans `projet/models/`
3. Lancer :
```bash
cd app/
streamlit run streamlit_app.py
```

Accès : `http://localhost:8501`

**📖 Guide détaillé** : Voir [WORKFLOW_COMPLET.md](WORKFLOW_COMPLET.md)

---

## 👥 Travail en Équipe (3 Membres)

Voir **[README_EQUIPE.md](README_EQUIPE.md)** pour l'organisation détaillée.

**Répartition recommandée :**
- **Personne 1** : Notebook_0 (données) + Notebook_1 (LSTM)
- **Personne 2** : Notebook_2 (BiLSTM) + Notebook_3 (CNN-BiLSTM)
- **Personne 3** : Notebook_4 (BERT avec GPU)
- **Tous ensemble** : Notebook_5 (comparaison)

---

## 📊 Modèles Implémentés

| Modèle | Architecture | Paramètres | Temps |
|--------|-------------|------------|-------|
| LSTM | LSTM simple (baseline) | ~500K | 15-20 min |
| BiLSTM + Attention | BiLSTM + Attention mechanism | ~800K | 30-40 min |
| CNN-BiLSTM + Attention | CNN + BiLSTM + Attention | ~1M | 40-50 min |
| BERT | Fine-tuning BERT-base-uncased | ~110M | 50-60 min |

**Temps total d'entraînement** : ~2h30-3h

---

## 🎨 Interface Streamlit

### Fonctionnalités
✅ Sélection des 4 modèles  
✅ Prédiction en temps réel  
✅ Visualisations interactives (barres, radar chart)  
✅ Top 10 émotions avec probabilités  
✅ Distribution des 28 émotions  
✅ Exemples prédéfinis

### Captures d'écran
Voir **[INTERFACE_STREAMLIT.md](notebooks/INTERFACE_STREAMLIT.md)** pour le guide complet.

---

## 📈 Dataset - GoEmotions

- **Source** : Reddit comments
- **Taille** : 58,000 commentaires
- **Classes** : 28 émotions + neutral
- **Type** : Multi-label classification
- **Split** : Train/Val/Test

**27 Émotions** : admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise + neutral

---

## 🎓 Évaluation

### Points clés
✅ 4 modèles de Deep Learning  
✅ Analyse comparative complète  
✅ Étude d'ablation  
✅ Interface interactive (+10-15% bonus)  
✅ Rapport détaillé avec résultats

---

## 📞 Support

Pour toute question sur l'organisation du travail en équipe ou l'interface, consulter :
- **[README_EQUIPE.md](README_EQUIPE.md)** : Organisation, workflow, timeline
- **[INTERFACE_STREAMLIT.md](notebooks/INTERFACE_STREAMLIT.md)** : Guide complet de l'interface

---

## 📝 Licence
Projet académique - 3ING - Indexation et Recherche d'Information
