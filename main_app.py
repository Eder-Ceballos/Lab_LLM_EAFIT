import streamlit as st
import os
import numpy as np
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(
    page_title="Plataforma LLM & NLP (Streamlit Cloud)",
    page_icon="🤖",
    layout="wide"
)

# --- BARRA LATERAL: CONFIGURACIÓN Y MODELOS ---
st.sidebar.header("🔑 Configuración de Groq")

# Intentar cargar desde st.secrets si está desplegado en Streamlit Cloud, si no, pedirlo en la barra lateral
groq_api_key = st.secrets.get("GROQ_API_KEY", "") if "GROQ_API_KEY" in st.secrets else ""
api_key_input = st.sidebar.text_input("Ingresa tu API Key de Groq (gsk_...)", value=groq_api_key, type="password")

if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input

st.sidebar.divider()
st.sidebar.header("⚙️ Parámetros del Modelo")

# Modelos estables y gratuitos en la capa free de Groq
# Modelos activos y estables en la API de Groq
model_options = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "gpt-oss-20b"
]
selected_model = st.sidebar.selectbox("Selecciona el LLM", model_options)

temperature = st.sidebar.slider("Temperatura", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider("Tokens Máximos", min_value=50, max_value=4096, value=512, step=50)
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

# --- CUERPO PRINCIPAL ---
st.title("🚀 Plataforma Interactiva de NLP y LLMs en la Nube")
st.markdown("Desplegado en **Streamlit Cloud** usando la velocidad y capa gratuita de **Groq**.")

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Generación de Texto", 
    "🔤 Tokenización y IDs", 
    "📊 Bag of Words & Similitud", 
    "📐 Embeddings"
])

# ==========================================
# PESTAÑA 1: GENERACIÓN DE TEXTO
# ==========================================
with tab1:
    st.header("Generación de Texto con Groq")
    prompt = st.text_area("Ingresa tu instrucción o prompt:", "Explica brevemente qué es la computación cuántica.")
    
    if st.button("Generar Respuesta", type="primary"):
        if not api_key_input:
            st.error("Por favor, ingresa tu API Key de Groq (gsk_...) en la barra lateral o configúrala en los Secrets de Streamlit.")
        else:
            try:
                client = Groq(api_key=api_key_input)
                with st.spinner("Generando respuesta en la nube..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    response_text = chat_completion.choices[0].message.content
                    
                    st.subheader("Respuesta del Modelo:")
                    st.write(response_text)
                    
                    if hasattr(chat_completion, 'usage') and chat_completion.usage:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Tokens de Prompt", chat_completion.usage.prompt_tokens)
                        col2.metric("Tokens de Respuesta", chat_completion.usage.completion_tokens)
                        col3.metric("Tokens Totales", chat_completion.usage.total_tokens)
            except Exception as e:
                st.error(f"Ocurrió un error al conectar con la API de Groq: {e}")

# ==========================================
# PESTAÑA 2: TOKENIZACIÓN Y TOKENS ID
# ==========================================
with tab2:
    st.header("Análisis de Tokens y Tokens ID")
    token_text = st.text_input("Texto a tokenizar:", "¡Hola! La inteligencia artificial está transformando el mundo.")
    
    if token_text:
        words = nltk.word_tokenize(token_text)
        vocab = {word: idx + 1000 for idx, word in enumerate(sorted(list(set(words))))}
        token_ids = [vocab[w] for w in words]
        
        df_tokens = pd.DataFrame({
            "Token": words,
            "Token ID (Simulado)": token_ids
        })
        st.dataframe(df_tokens, use_container_width=True)
        st.info(f"**Conteo total de tokens:** {len(words)}")

# ==========================================
# PESTAÑA 3: BAG OF WORDS Y SIMILITUD
# ==========================================
with tab3:
    st.header("Bolsa de Palabras (BoW) y Métricas de Similitud")
    col_a, col_b = st.columns(2)
    with col_a:
        doc1 = st.text_area("Documento 1:", "El aprendizaje automático es una rama de la inteligencia artificial.")
    with col_b:
        doc2 = st.text_area("Documento 2:", "La inteligencia artificial y el machine learning son tecnologías avanzadas.")
    
    if st.button("Calcular Similitud y BoW"):
        vectorizer = CountVectorizer()
        try:
            bow_matrix = vectorizer.fit_transform([doc1, doc2])
            feature_names = vectorizer.get_feature_names_out()
            df_bow = pd.DataFrame(bow_matrix.toarray(), columns=feature_names, index=["Doc 1", "Doc 2"])
            
            st.subheader("Matriz Bolsa de Palabras (BoW)")
            st.dataframe(df_bow, use_container_width=True)
            
            cosine_sim = cosine_similarity(bow_matrix[0:1], bow_matrix[1:2])[0][0]
            st.subheader("Métrica de Similitud")
            st.metric("Similitud de Coseno", f"{cosine_sim:.4f}")
        except Exception as e:
            st.warning("Por favor asegúrate de ingresar textos válidos.")

# ==========================================
# PESTAÑA 4: EMBEDDINGS
# ==========================================
with tab4:
    st.header("Generación de Embeddings")
    embed_text = st.text_input("Texto para embedding:", "Los modelos de lenguaje grande procesan información en espacios vectoriales.")
    
    if st.button("Generar Embedding"):
        try:
            from sentence_transformers import SentenceTransformer
            @st.cache_resource
            def load_embed_model():
                return SentenceTransformer('all-MiniLM-L6-v2')
            
            model = load_embed_model()
            embedding = model.encode(embed_text)
            
            st.success("¡Embedding generado exitosamente!")
            st.write(f"**Dimensión del vector:** {len(embedding)}")
            st.write("**Primeros 10 valores del vector:**")
            st.write(embedding[:10])
            
            st.bar_chart(embedding[:50])
            st.caption("Visualización de los primeros 50 componentes del embedding.")
        except Exception as e:
            st.error(f"Error al cargar el modelo de embeddings: {e}")
