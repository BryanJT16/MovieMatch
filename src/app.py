import streamlit as st
import pandas as pd
import os
import gc
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np    
import ast
import requests
import urllib.parse

api_key = "fa1b7162"  

# Configuración de la página
st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="wide"
)

st.title("Sistema de Recomendación de Películas")
st.markdown("### Encuentra tu próxima película favorita")

# ==========================================
# 1. SISTEMA DE CARGA OPTIMIZADO CON CACHÉ
# ==========================================

@st.cache_resource
def cargar_modelo():
    try:
        dir_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        if os.path.basename(dir_actual) != "src":
            ruta_modelo = os.path.join(dir_actual, "src", "model", "rotten_pipeline.pkl")
        else:
            ruta_modelo = os.path.join(dir_actual, "model", "rotten_pipeline.pkl")
            
        with open(ruta_modelo, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    except Exception as e:
        st.error(f"Error al cargar el modelo (.pkl): {str(e)}")
        return None

@st.cache_data
def cargar_dataset():
    try:
        dir_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        if os.path.basename(dir_actual) != "src":
            ruta_movies = os.path.join(dir_actual, "src", "dataset", "movies.csv")
        else:
            ruta_movies = os.path.join(dir_actual, "dataset", "movies.csv")
            
        columnas_necesarias = [
            'movie_title', 'genres', 'actors', 'directors', 
            'tomatometer_rating', 'movie_info', 'critics_consensus', 
            'release_year'
        ]
        movies = pd.read_csv(ruta_movies, usecols=columnas_necesarias)
        
        # Limpieza inmediata en caché (Solo se hace una vez al cargar la app)
        movies["genres"] = movies["genres"].fillna("")
        movies["actors"] = movies["actors"].fillna("")
        movies["directors"] = movies["directors"].fillna("")
        movies["movie_info"] = movies["movie_info"].fillna("")
        movies["critics_consensus"] = movies["critics_consensus"].fillna("")
        movies["release_year"] = pd.to_numeric(movies["release_year"], errors='coerce').fillna(0).astype(int)
        movies["tomatometer_rating"] = pd.to_numeric(movies["tomatometer_rating"], errors='coerce').fillna(0).astype(int)
        
        return movies
    except Exception as e:
        st.error(f"Error al cargar las películas (.csv): {str(e)}")
        return None

pipeline = cargar_modelo()
movies = cargar_dataset()

# ==========================================
# 2. CACHEAR EL PROCESAMIENTO DE FILTROS (¡Clave para evitar OOM!)
# ==========================================
@st.cache_data
def procesar_filtros_unicos(df):
    # Procesar géneros (¡DESCOMENTADO Y OPTIMIZADO EN CACHÉ!)
    all_genres = []
    for genres in df["genres"].dropna():
        genres_str = genres.replace("[","").replace("]","").replace("'","")
        genres_list = [g.strip() for g in genres_str.split(",") if g.strip()]
        all_genres.extend(genres_list)
    unique_genres = sorted(list(set([g.strip() for g in all_genres if g])))

    # Procesar actores (Tu algoritmo original optimizado)
    all_actors = set()
    for actors_str in df["actors"].dropna():
        actors_str = actors_str.replace("[", "").replace("]", "").replace("'", "")
        actors = [actor.strip() for actor in actors_str.split(",") if actor.strip()]
        all_actors.update(actors)
    unique_actors = sorted(list(all_actors))
    unique_actors.insert(0, "Todos")

    # Procesar directores (Tu algoritmo original optimizado)
    all_directors = set()
    for directors_str in df["directors"].dropna():
        directors_str = directors_str.replace("[", "").replace("]", "").replace("'", "")
        directors = [director.strip() for director in directors_str.split(",") if director.strip()]
        all_directors.update(directors)
    unique_directors = sorted(list(all_directors))
    unique_directors.insert(0, "Todos")

    # Años
    years_clean = pd.to_numeric(df["release_year"], errors='coerce').dropna().astype(int)
    years_without_zeros = years_clean[years_clean > 0]
    min_year = int(years_without_zeros.min()) if not years_without_zeros.empty else 1900
    max_year = int(years_without_zeros.max()) if not years_without_zeros.empty else 2026

    return unique_genres, unique_actors, unique_directors, min_year, max_year


def obtener_poster(titulo):
    try:
        titulo_encoded = urllib.parse.quote(titulo)  
        url = f"http://www.omdbapi.com/?t={titulo_encoded}&apikey={api_key}"
        response = requests.get(url, timeout=3) # bajado a 3 segundos de timeout para no congelar hilos
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True" and "Poster" in data:
                return data["Poster"]
        return None
    except Exception:
        return None


# ==========================================
# 4. CONSTRUCCIÓN DE LA INTERFAZ
# ==========================================
if pipeline is not None and movies is not None:
    
    if "filtros_precargados" not in st.session_state:
        # Esto SOLO se ejecutará la primerísima vez que el usuario abra la página
        unique_genres, unique_actors, unique_directors, min_year, max_year = procesar_filtros_unicos(movies)
        
        st.session_state["filtros_precargados"] = {
            "genres": unique_genres,
            "actors": unique_actors,
            "directors": unique_directors,
            "min_year": min_year,
            "max_year": max_year
        }
    
    # Extraemos los filtros directamente de la sesión (es instantáneo y consume 0% de CPU/RAM de proceso)
    filtros = st.session_state["filtros_precargados"]
    
    # Sidebar de filtros utilizando los datos de la sesión
    st.sidebar.header("Filtros de Búsqueda")
    
    selected_genres = st.sidebar.multiselect("Géneros", filtros["genres"])
    
    year_range = st.sidebar.slider(
        "Rango de Años",
        min_value=filtros["min_year"],
        max_value=filtros["max_year"],
        value=(filtros["min_year"], filtros["max_year"])
    )
    
    selected_actor = st.sidebar.selectbox(
        "Selecciona un Actor/Actriz:",
        options=filtros["actors"],
        index=0,
        help="Escribe para buscar un actor o actriz del reparto."
    )
    selected_director = st.sidebar.selectbox(
        "Selecciona un Director:",
        options=filtros["directors"],
        index=0,
        help="Escribe para buscar al director de tu preferencia."
    )
    
    selected_tomatometer = st.sidebar.slider("Puntuación mínima en Tomatometer", 0, 100, 50)

    # Película de referencia
    lista_peliculas = sorted(movies["movie_title"].unique())
    selected_movie = st.selectbox(
        "🔍 Escribe o selecciona una película de referencia:",
        options=lista_peliculas,
        index=0,
        help="Empieza a escribir el nombre de la película que te guste para buscarla."
    )
    
    # Pesos para la recomendación
    st.sidebar.header("Ajustes del Modelo")
    alpha = st.sidebar.slider("Peso de la película seleccionada", 0.0, 1.0, 0.7)
    beta = st.sidebar.slider("Peso de los filtros", 0.0, 1.0, 0.2)
    gamma = st.sidebar.slider("Peso del sentimiento del crítico", 0.0, 1.0, 0.1)
    
    top_n = st.sidebar.number_input("Número de recomendaciones", 1, 20, 5)

    # ==========================================
    # 5. MOTOR DE RECOMENDACIÓN OPTIMIZADO
    # ==========================================
    def recomendar_peliculas():
        
        # Trabajamos sobre una vista temporal filtrada o el original sin duplicar RAM de manera excesiva
        # Rellenamos nulos sobre la marcha de forma eficiente
        movies["genres"] = movies["genres"].fillna("")
        movies["actors"] = movies["actors"].fillna("")
        movies["directors"] = movies["directors"].fillna("")
        movies["movie_info"] = movies["movie_info"].fillna("")
        movies["critics_consensus"] = movies["critics_consensus"].fillna("")
        movies["release_year_clean"] = pd.to_numeric(movies["release_year"], errors='coerce').fillna(0)
        movies["tomatometer_rating_clean"] = movies["tomatometer_rating"].fillna(0)

        # Calcular match_score de manera limpia
        match_score = np.zeros(len(movies))
        
        if selected_genres:
            match_score += movies["genres"].apply(lambda x: sum(g in x for g in selected_genres))
        if selected_actor and selected_actor != "Todos":
            match_score += movies["actors"].apply(lambda x: 2 if selected_actor in x else 0)
        if selected_director and selected_director != "Todos":
            match_score += movies["directors"].apply(lambda x: 2 if selected_director in x else 0)
            
        match_score += movies["release_year_clean"].apply(
            lambda x: 1 if year_range[0] <= int(x) <= year_range[1] else 0
        )
        match_score += movies["tomatometer_rating_clean"].apply(
            lambda x: 1 if x >= selected_tomatometer else 0
        )

        # Normalizar match_score evitando divisiones por cero
        max_match = match_score.max()
        match_norm = match_score / max_match if max_match > 0 else match_score

        # Calcular sentimiento
        consensus_sentiment_prob = pipeline.predict_proba(movies["critics_consensus"])[:, 1]
        max_sentiment = consensus_sentiment_prob.max()
        consensus_sentiment_norm = (
            consensus_sentiment_prob / max_sentiment if max_sentiment > 0 else consensus_sentiment_prob
        )

        # Crear matriz TF-IDF (Bajado max_features a 1000 para blindar los 512 MB de Render)
        combined_features = (
            movies["genres"] + " " +
            movies["directors"] + " " +
            movies["actors"] + " " +
            movies["movie_info"]
        )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(combined_features)

        if not selected_movie:
            final_score = (
                beta * match_norm + 
                gamma * consensus_sentiment_norm
            )
            top_indices = final_score.argsort()[::-1][:top_n]
        else:
            idx = movies[movies["movie_title"] == selected_movie].index[0]
            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            final_score = (
                alpha * cosine_sim +
                beta * match_norm +
                gamma * consensus_sentiment_norm
            )
            
            top_indices = final_score.argsort()[::-1]
            top_indices = [i for i in top_indices if i != idx]
            top_indices = top_indices[:top_n]

        # Guardar temporalmente las métricas que usaremos para pintar
        resultado = movies.iloc[top_indices].copy()
        resultado["consensus_sentiment_norm"] = consensus_sentiment_norm[top_indices]
        
        # Liberar memoria de variables del cálculo matemático de inmediato
        del tfidf_matrix
        del combined_features
        del cosine_sim
        gc.collect()
        
        return resultado

    # Botón para generar recomendaciones
    if st.button("Obtener Recomendaciones"):
        with st.spinner("Generando recomendaciones..."):
            recommendations = recomendar_peliculas()
            
            # Obtener posters
            posters = [obtener_poster(t) for t in recommendations["movie_title"].tolist()]
            
            # Mostrar recomendaciones
            for (idx, movie), poster_url in zip(recommendations.iterrows(), posters):
                
                # Gestión segura de listas vacías o mal formateadas
                try:
                    genres = ast.literal_eval(movie['genres']) if movie['genres'] else []
                except Exception:
                    genres = [g.strip() for g in movie['genres'].replace("[","").replace("]","").replace("'","").split(",")]
                    
                try:
                    directors = ast.literal_eval(movie['directors']) if movie['directors'] else []
                except Exception:
                    directors = [d.strip() for d in movie['directors'].replace("[","").replace("]","").replace("'","").split(",")]
                    
                try:
                    actors = ast.literal_eval(movie['actors']) if movie['actors'] else []
                except Exception:
                    actors = [a.strip() for a in movie['actors'].replace("[","").replace("]","").replace("'","").split(",")]

                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster_url and poster_url != "N/A":
                            st.image(poster_url, use_container_width=True)
                        else:
                            placeholder_url = "https://dummyimage.com/200x300/2e2e2e/ffffff.png&text=No+Image"
                            st.image(placeholder_url, use_container_width=True)

                        metric_col1, metric_col2 = st.columns(2)
                        metric_col1.metric("Tomatometer", f"{int(movie['tomatometer_rating'])}%")
                        metric_col2.metric("Sentimiento", f"{movie['consensus_sentiment_norm']:.2f}")
                
                    with col2:
                        st.subheader(movie["movie_title"])
                        st.write(f"**Géneros:** {', '.join(genres)}")
                        st.write(f"**Director:** {', '.join(directors)}")
                        st.write(f"**Actores principales:** {', '.join(actors[:5])}...")
                        if movie['movie_info']:
                            st.write(f"**Información:** {movie['movie_info']}")
                        if movie['critics_consensus']:
                            st.write(f"**Consenso de críticos:** {movie['critics_consensus']}")
                st.divider()
            
            # Limpiar memoria residual tras pintar las cards de películas
            gc.collect()
else:
    st.error("No se pudieron cargar los datos necesarios para la aplicación.")
    st.info("Por favor, verifica que los archivos de datos y el modelo estén disponibles en las rutas correctas.")