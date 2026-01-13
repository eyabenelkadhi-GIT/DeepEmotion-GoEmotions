"""
Interface Streamlit pour la Détection d'Émotions
Projet GoEmotions - 28 Émotions
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import plotly.graph_objects as go
import plotly.express as px
import re
import os

# Configuration de la page
st.set_page_config(
    page_title="Détection d'Émotions - GoEmotions",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Liste des 28 émotions
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

MAX_SEQUENCE_LENGTH = 128

# Fonction de nettoyage du texte
def clean_text(text):
    """Nettoie le texte comme dans le preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Chargement des modèles
@st.cache_resource
def load_tokenizers():
    """Charge les tokenizers (Keras et BERT)"""
    tokenizers = {}
    
    # 1. Keras Tokenizer (pour LSTM, BiLSTM, CNN)
    try:
        paths = [
            '../data/processed/tokenizer.pkl',
            'data/processed/tokenizer.pkl',
            'tokenizer.pkl'
        ]
        for path in paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    tokenizers['keras'] = pickle.load(f)
                break
        if 'keras' not in tokenizers:
            st.warning("⚠️ Tokenizer Keras non trouvé.")
    except Exception as e:
        st.error(f"Erreur loading Keras tokenizer: {e}")

    # 2. BERT Tokenizer
    try:
        from transformers import BertTokenizer
        tokenizers['bert'] = BertTokenizer.from_pretrained('bert-base-uncased')
    except Exception as e:
        st.error(f"Erreur loading BERT tokenizer: {e}")
        
    return tokenizers

@st.cache_resource
def load_model(model_name):
    """Charge le modèle sélectionné"""
    # Chemins ajustés selon les notebooks
    model_paths = {
        'LSTM': ['../models/lstm/best_model.h5', 'models/lstm/best_model.h5'],
        'BiLSTM + Attention': ['../models/bilstm/best_model.h5', 'models/bilstm/best_model.h5'],
        'CNN-BiLSTM + Attention': ['../models/cnn_bilstm/best_model.h5', 'models/cnn_bilstm/best_model.h5'],
        'BERT': ['../models/bert/best_model', 'models/bert/best_model'] # SavedModel format (folder)
    }
    
    # Essayer de trouver le bon chemin
    selected_path = None
    if model_name in model_paths:
        for path in model_paths[model_name]:
            if os.path.exists(path):
                selected_path = path
                break
    
    if not selected_path:
        return None

    try:
        # Custom objects pour les couches personnalisées
        custom_objects = {}
        if 'Attention' in model_name:
            # Définir AttentionLayer si nécessaire (copie de la classe des notebooks)
            class AttentionLayer(keras.layers.Layer):
                def __init__(self, **kwargs):
                    super(AttentionLayer, self).__init__(**kwargs)
                def build(self, input_shape):
                    self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
                    self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
                    super(AttentionLayer, self).build(input_shape)
                def call(self, x):
                    e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
                    a = keras.backend.softmax(e, axis=1)
                    output = x * a
                    return keras.backend.sum(output, axis=1)
                def get_config(self):
                    return super(AttentionLayer, self).get_config()
            
            custom_objects['AttentionLayer'] = AttentionLayer

        # Chargement
        if model_name == 'BERT':
            # BERT est souvent sauvegardé en TF SavedModel
            from transformers import TFBertModel
            custom_objects['TFBertModel'] = TFBertModel
            model = keras.models.load_model(selected_path, custom_objects=custom_objects)
        else:
            model = keras.models.load_model(selected_path, custom_objects=custom_objects)
            
        return model
    except Exception as e:
        st.error(f"❌ Erreur détailée chargement {model_name}: {e}")
        return None

# Fonction de prédiction
def predict_emotions(text, model, tokenizers, model_name, top_k=10):
    """Prédit les émotions pour un texte donné"""
    cleaned_text = clean_text(text)
    
    # Préparation spécifique selon le modèle
    if model_name == 'BERT':
        tokenizer = tokenizers['bert']
        encoding = tokenizer(
            cleaned_text,
            max_length=MAX_SEQUENCE_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        inputs = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
        predictions = model.predict(inputs, verbose=0)[0] # BERT retourne souvent un tuple ou dict
        # Si le modèle retourne un objet TF, il faut extraire les logits/probs
        if isinstance(predictions, dict):
             predictions = predictions['output'] # Ajuster selon le nom de la couche de sortie
    else:
        # Modèles Keras standards
        tokenizer = tokenizers['keras']
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded = keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=MAX_SEQUENCE_LENGTH, 
            padding='post'
        )
        predictions = model.predict(padded, verbose=0)[0]
    
    # Top K émotions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_emotions = [EMOTIONS[i] for i in top_indices]
    top_probs = [predictions[i] for i in top_indices]
    
    return top_emotions, top_probs, predictions

# Interface principale
def main():
    # Titre et description
    st.title("🎭 Détecteur d'Émotions - GoEmotions Dataset")
    st.markdown("""
    Cette application utilise des modèles de Deep Learning pour détecter **28 émotions** 
    différentes dans un texte. Basée sur le dataset **GoEmotions** (58,000 commentaires Reddit).
    """)
    
    # Sidebar - Configuration
    st.sidebar.title("⚙️ Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Sélectionner un modèle",
        ["LSTM", "BiLSTM + Attention", "CNN-BiLSTM + Attention", "BERT"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informations du Modèle")
    
    # Informations sur les modèles
    model_info = {
        "LSTM": {
            "description": "Modèle baseline avec LSTM simple",
            "params": "~500K paramètres",
            "temps": "~15-20 min"
        },
        "BiLSTM + Attention": {
            "description": "BiLSTM avec mécanisme d'attention",
            "params": "~800K paramètres",
            "temps": "~30-40 min"
        },
        "CNN-BiLSTM + Attention": {
            "description": "Architecture hybride CNN + BiLSTM",
            "params": "~1M paramètres",
            "temps": "~40-50 min"
        },
        "BERT": {
            "description": "Fine-tuning BERT-base-uncased",
            "params": "~110M paramètres",
            "temps": "~50-60 min"
        }
    }
    
    st.sidebar.info(f"**{model_choice}**\n\n{model_info[model_choice]['description']}")
    st.sidebar.caption(f"⚡ {model_info[model_choice]['params']}")
    
    # Chargement des ressources
    with st.spinner("Chargement des tokenizers..."):
        tokenizers = load_tokenizers()
    
    with st.spinner(f"Chargement du modèle {model_choice}..."):
        model = load_model(model_choice)
    
    if model is None:
        st.error("❌ Impossible de charger le modèle. Vérifiez que les modèles sont entraînés.")
        st.info("💡 **Instructions** : Exécutez d'abord les notebooks d'entraînement (Notebook_1 à Notebook_4)")
        return
    
    st.success(f"✅ Modèle {model_choice} chargé avec succès!")
    
    # Zone de saisie
    st.markdown("---")
    st.header("✍️ Entrez votre texte")
    
    # Exemples prédéfinis
    examples = {
        "Exemple 1 (Joie)": "I'm so happy and excited about this amazing news!",
        "Exemple 2 (Colère)": "This is absolutely frustrating and makes me so angry!",
        "Exemple 3 (Tristesse)": "I feel so sad and disappointed about what happened.",
        "Exemple 4 (Surprise)": "Wow, I can't believe this happened! What a surprise!",
        "Exemple 5 (Peur)": "I'm really scared and nervous about the situation."
    }
    
    selected_example = st.selectbox("Ou choisir un exemple", ["---"] + list(examples.keys()))
    
    if selected_example != "---":
        default_text = examples[selected_example]
    else:
        default_text = ""
    
    user_input = st.text_area(
        "Texte à analyser",
        value=default_text,
        height=150,
        placeholder="Entrez un texte en anglais (ex: I love this wonderful day!)",
        help="Le modèle fonctionne mieux avec du texte en anglais"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyser", type="primary")
    with col2:
        clear_button = st.button("🗑️ Effacer")
    
    if clear_button:
        st.rerun()
    
    # Analyse
    if analyze_button and user_input.strip():
        with st.spinner("Analyse en cours..."):
            top_emotions, top_probs, all_predictions = predict_emotions(
                user_input, model, tokenizers, model_choice, top_k=10
            )
        
        # Résultats
        st.markdown("---")
        st.header("📊 Résultats de l'Analyse")
        
        # Métriques principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "🥇 Émotion Principale", 
                top_emotions[0].capitalize(),
                f"{top_probs[0]:.1%}"
            )
        with col2:
            st.metric(
                "🥈 Deuxième Émotion", 
                top_emotions[1].capitalize(),
                f"{top_probs[1]:.1%}"
            )
        with col3:
            st.metric(
                "🥉 Troisième Émotion", 
                top_emotions[2].capitalize(),
                f"{top_probs[2]:.1%}"
            )
        
        # Visualisations
        tab1, tab2, tab3 = st.tabs(["📊 Top 10 Émotions", "🎯 Toutes les Émotions", "📈 Distribution"])
        
        with tab1:
            # Graphique en barres - Top 10
            fig = go.Figure(data=[
                go.Bar(
                    x=top_probs,
                    y=[e.capitalize() for e in top_emotions],
                    orientation='h',
                    marker=dict(
                        color=top_probs,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Probabilité")
                    ),
                    text=[f"{p:.1%}" for p in top_probs],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Top 10 Émotions Détectées",
                xaxis_title="Probabilité",
                yaxis_title="Émotion",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Toutes les émotions
            emotions_df = pd.DataFrame({
                'Émotion': EMOTIONS,
                'Probabilité': all_predictions
            }).sort_values('Probabilité', ascending=False)
            
            fig = px.bar(
                emotions_df,
                x='Probabilité',
                y='Émotion',
                orientation='h',
                color='Probabilité',
                color_continuous_scale='RdYlGn',
                title="Distribution Complète des 28 Émotions"
            )
            
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=top_probs[:8],
                theta=[e.capitalize() for e in top_emotions[:8]],
                fill='toself',
                name='Émotions'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(top_probs)]
                    )
                ),
                showlegend=False,
                title="Radar Chart - Top 8 Émotions",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        st.markdown("---")
        st.subheader("📋 Détails des Prédictions")
        
        results_df = pd.DataFrame({
            'Rang': range(1, 11),
            'Émotion': [e.capitalize() for e in top_emotions],
            'Probabilité': [f"{p:.4f}" for p in top_probs],
            'Pourcentage': [f"{p:.2%}" for p in top_probs]
        })
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Texte nettoyé
        with st.expander("🧹 Voir le texte nettoyé"):
            st.code(clean_text(user_input))
    
    elif analyze_button:
        st.warning("⚠️ Veuillez entrer un texte à analyser.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🎓 Projet Détection d'Émotions - Deep Learning</p>
        <p>📚 Dataset: GoEmotions (58,000 commentaires Reddit, 28 émotions)</p>
        <p>🏛️ 3ING - Indexation et Recherche d'Information</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
