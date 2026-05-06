import streamlit as st
import pandas as pd
import os
import pickle
import requests
import urllib.parse
import numpy as np
import gc      # Mantenemos esto para estabilidad interna

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

api_key = "fa1b7162"

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Sistema de Recomendación de Películas")
st.markdown("### Encuentra tu próxima película favorita")

# ==========================================
# RUTAS DE ARCHIVOS
# ==========================================

def get_base_dir():
    current = (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals()
        else os.getcwd()
    )
    if os.path.basename(current) != "src":
        return os.path.join(current, "src")
    return current

BASE_DIR = get_base_dir()

# ==========================================
# CARGA DE RECURSOS (OPTIMIZADO)
# ==========================================

@st.cache_resource
def cargar_modelo():
    path = os.path.join(BASE_DIR, "model", "rotten_pipeline.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def cargar_dataset():
    path = os.path.join(BASE_DIR, "dataset", "movies.csv")
    cols = [
        "movie_title", "genres", "actors", "directors",
        "tomatometer_rating", "movie_info", "critics_consensus", "release_year"
    ]
    df = pd.read_csv(path, usecols=cols)
    
    text_cols = ["genres", "actors", "directors", "movie_info", "critics_consensus"]
    for col in text_cols:
        df[col] = df[col].fillna("")

    # Tipos de datos optimizados para Render
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(np.int16)
    df["tomatometer_rating"] = pd.to_numeric(df["tomatometer_rating"], errors="coerce").fillna(0).astype(np.int16)
    
    gc.collect()
    return df

@st.cache_resource
def cargar_tfidf(df):
    features = df["genres"] + " " + df["directors"]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=150,
        ngram_range=(1, 1)
    )
    matrix = vectorizer.fit_transform(features)
    gc.collect()
    return matrix

@st.cache_resource
def cargar_sentimientos(_pipeline, _df):
    probs = _pipeline.predict_proba(_df["critics_consensus"])[:, 1]
    gc.collect()
    return probs

# ==========================================
# DATOS DE INTERFAZ Y PÓSTERS
# ==========================================

@st.cache_data
def preparar_ui(df):
    titles = sorted(df["movie_title"].unique())
    genres = sorted(set(
        g.strip()
        for row in df["genres"]
        for g in row.replace("[", "").replace("]", "").replace("'", "").split(",")
        if g.strip()
    ))
    years = df["release_year"]
    years = years[years > 0]
    return titles, genres, int(years.min()), int(years.max())

@st.cache_data(ttl=3600)
def obtener_poster(title):
    try:
        title_enc = urllib.parse.quote(title)
        url = f"http://www.omdbapi.com/?t={title_enc}&apikey={api_key}"
        r = requests.get(url, timeout=2)
        data = r.json()
        if data.get("Response") == "True":
            return data.get("Poster")
    except:
        pass
    return None

# ==========================================
# INICIALIZACIÓN
# ==========================================

pipeline = cargar_modelo()
movies = cargar_dataset()
tfidf_matrix = cargar_tfidf(movies)
sentiment_probs = cargar_sentimientos(pipeline, movies)

(movie_titles, unique_genres, min_year, max_year) = preparar_ui(movies)

# ==========================================
# BARRA LATERAL (FILTROS)
# ==========================================

st.sidebar.header("Filtros")
selected_genres = st.sidebar.multiselect("Géneros", unique_genres)
year_range = st.sidebar.slider("Años", min_year, max_year, (min_year, max_year))
selected_tomatometer = st.sidebar.slider("Tomatometer mínimo", 0, 100, 50)

st.sidebar.header("Pesos de Recomendación")
alpha = st.sidebar.slider("Importancia Película", 0.0, 1.0, 0.7)
beta = st.sidebar.slider("Importancia Filtros", 0.0, 1.0, 0.2)
gamma = st.sidebar.slider("Importancia Crítica", 0.0, 1.0, 0.1)

top_n = st.sidebar.number_input("Resultados a mostrar", 1, 20, 5)

selected_movie = st.selectbox("Basado en la película:", movie_titles)

# ==========================================
# LÓGICA DE RECOMENDACIÓN
# ==========================================

def recomendar():
    match_score = np.zeros(len(movies), dtype=np.float32)

    if selected_genres:
        match_score += movies["genres"].apply(lambda x: sum(g in x for g in selected_genres))

    match_score += np.where((movies["release_year"] >= year_range[0]) & 
                            (movies["release_year"] <= year_range[1]), 0.5, 0)
    match_score += np.where(movies["tomatometer_rating"] >= selected_tomatometer, 0.5, 0)

    if match_score.max() > 0:
        match_score /= match_score.max()

    sentiment_norm = sentiment_probs / sentiment_probs.max() if sentiment_probs.max() > 0 else sentiment_probs

    idx = movies[movies["movie_title"] == selected_movie].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    final_score = (alpha * cosine_sim) + (beta * match_score) + (gamma * sentiment_norm)

    indices = final_score.argsort()[::-1]
    indices = [i for i in indices if i != idx][:top_n]

    result = movies.iloc[indices].copy()
    result["sentiment"] = sentiment_norm[indices]
    
    gc.collect()
    return result

# ==========================================
# RESULTADOS
# ==========================================

if st.button("🚀 Obtener Recomendaciones"):
    with st.spinner("Calculando..."):
        recommendations = recomendar()
        
        for _, movie in recommendations.iterrows():
            poster = obtener_poster(movie["movie_title"])
            col1, col2 = st.columns([1, 3])

            with col1:
                # Lógica de póster con imagen de sustitución
                if poster and poster != "N/A":
                    st.image(poster, use_container_width=True)
                else:
                    placeholder = "https://dummyimage.com/200x300/2e2e2e/ffffff.png&text=Sin+Poster"
                    st.image(placeholder, use_container_width=True)
                
                m1, m2 = st.columns(2)
                m1.metric("Tomatometer", f"{int(movie['tomatometer_rating'])}%")
                m2.metric("Sentimiento", f"{movie['sentiment']:.2f}")

            with col2:
                st.subheader(movie["movie_title"])
                if movie["movie_info"]:
                    st.write(f"**Sinopsis:** {movie['movie_info']}")
                if movie["critics_consensus"]:
                    st.write(f"**Consenso Crítico:** {movie['critics_consensus']}")
            st.divider()
        
        gc.collect()