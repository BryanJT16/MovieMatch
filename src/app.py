import streamlit as st
import pandas as pd
import os
import psutil
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import requests
import urllib.parse


api_key = "fa1b7162"


# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Sistema de Recomendación de Películas")
st.markdown("### Encuentra tu próxima película favorita")


# ==========================================
# MEMORIA
# ==========================================

def medir_ram(etapa=""):
    proceso = psutil.Process(os.getpid())
    ram_mb = proceso.memory_info().rss / (1024 * 1024)
    st.sidebar.write(f"📊 **RAM en {etapa}:** {ram_mb:.2f} MB")


st.sidebar.markdown("### 🖥️ Diagnóstico de Memoria")
medir_ram("Inicio App")


# ==========================================
# PATHS
# ==========================================

def get_base_dir():
    dir_actual = (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in locals()
        else os.getcwd()
    )

    if os.path.basename(dir_actual) != "src":
        return os.path.join(dir_actual, "src")

    return dir_actual


BASE_DIR = get_base_dir()


# ==========================================
# CARGA MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():

    ruta_modelo = os.path.join(
        BASE_DIR,
        "model",
        "rotten_pipeline.pkl"
    )

    with open(ruta_modelo, "rb") as f:
        return pickle.load(f)


# ==========================================
# CARGA DATASET
# ==========================================

@st.cache_data
def cargar_dataset():

    ruta_movies = os.path.join(
        BASE_DIR,
        "dataset",
        "movies.csv"
    )

    columnas = [
        'movie_title',
        'genres',
        'actors',
        'directors',
        'tomatometer_rating',
        'movie_info',
        'critics_consensus',
        'release_year'
    ]

    movies = pd.read_csv(
        ruta_movies,
        usecols=columnas
    )

    # Limpieza una sola vez
    string_cols = [
        "genres",
        "actors",
        "directors",
        "movie_info",
        "critics_consensus"
    ]

    for col in string_cols:
        movies[col] = movies[col].fillna("")

    movies["release_year"] = (
        pd.to_numeric(
            movies["release_year"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    movies["tomatometer_rating"] = (
        pd.to_numeric(
            movies["tomatometer_rating"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    return movies


pipeline = cargar_modelo()
movies = cargar_dataset()

medir_ram("Datos cargados")


# ==========================================
# TFIDF (SOLO UNA VEZ)
# ==========================================

@st.cache_resource
def cargar_tfidf(df):

    combined_features = (
        df["genres"]
        + " "
        + df["directors"]
        + " "
        + df["actors"]
        + " "
        + df["movie_info"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        combined_features
    )

    return tfidf_matrix


tfidf_matrix = cargar_tfidf(movies)

medir_ram("TFIDF cargado")


# ==========================================
# SENTIMIENTO (SOLO UNA VEZ)
# ==========================================

@st.cache_data
def cargar_sentimientos():

    return pipeline.predict_proba(
        movies["critics_consensus"]
    )[:, 1]


sentiment_probs = cargar_sentimientos()

medir_ram("Sentimientos cargados")


# ==========================================
# FILTROS
# ==========================================

@st.cache_data
def procesar_filtros(df):

    all_genres = []

    for genres in df["genres"]:

        genres_str = (
            genres
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
        )

        all_genres.extend(
            [
                g.strip()
                for g in genres_str.split(",")
                if g.strip()
            ]
        )

    unique_genres = sorted(
        list(set(all_genres))
    )

    unique_actors = set()
    unique_directors = set()

    for val in df["actors"]:
        val = (
            val
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
        )

        unique_actors.update(
            [
                x.strip()
                for x in val.split(",")
                if x.strip()
            ]
        )

    for val in df["directors"]:
        val = (
            val
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
        )

        unique_directors.update(
            [
                x.strip()
                for x in val.split(",")
                if x.strip()
            ]
        )

    years = movies["release_year"]
    years = years[years > 0]

    return (
        unique_genres,
        ["Todos"] + sorted(unique_actors),
        ["Todos"] + sorted(unique_directors),
        int(years.min()),
        int(years.max())
    )


(
    unique_genres,
    unique_actors,
    unique_directors,
    min_year,
    max_year
) = procesar_filtros(movies)


# ==========================================
# POSTERS CACHEADOS
# ==========================================

@st.cache_data
def obtener_poster(titulo):

    try:

        titulo_encoded = urllib.parse.quote(
            titulo
        )

        url = (
            f"http://www.omdbapi.com/"
            f"?t={titulo_encoded}"
            f"&apikey={api_key}"
        )

        response = requests.get(
            url,
            timeout=2
        )

        data = response.json()

        if data.get("Response") == "True":
            return data.get("Poster")

    except:
        pass

    return None


# ==========================================
# UI
# ==========================================

st.sidebar.header("Filtros")

selected_genres = st.sidebar.multiselect(
    "Géneros",
    unique_genres
)

year_range = st.sidebar.slider(
    "Años",
    min_year,
    max_year,
    (min_year, max_year)
)

selected_actor = st.sidebar.selectbox(
    "Actor",
    unique_actors
)

selected_director = st.sidebar.selectbox(
    "Director",
    unique_directors
)

selected_tomatometer = st.sidebar.slider(
    "Tomatometer mínimo",
    0,
    100,
    50
)

selected_movie = st.selectbox(
    "Película referencia",
    sorted(movies["movie_title"].unique())
)


st.sidebar.header("Pesos")

alpha = st.sidebar.slider(
    "Película",
    0.0,
    1.0,
    0.7
)

beta = st.sidebar.slider(
    "Filtros",
    0.0,
    1.0,
    0.2
)

gamma = st.sidebar.slider(
    "Crítica",
    0.0,
    1.0,
    0.1
)

top_n = st.sidebar.number_input(
    "Top N",
    1,
    20,
    5
)


# ==========================================
# RECOMENDADOR
# ==========================================

def recomendar():

    medir_ram("Inicio recomendación")

    match_score = np.zeros(
        len(movies)
    )

    if selected_genres:

        genre_mask = movies["genres"].apply(
            lambda x: sum(
                g in x
                for g in selected_genres
            )
        )

        match_score += genre_mask

    if selected_actor != "Todos":

        match_score += np.where(
            movies["actors"].str.contains(
                selected_actor,
                regex=False
            ),
            2,
            0
        )

    if selected_director != "Todos":

        match_score += np.where(
            movies["directors"].str.contains(
                selected_director,
                regex=False
            ),
            2,
            0
        )

    match_score += np.where(
        (
            movies["release_year"]
            >= year_range[0]
        )
        &
        (
            movies["release_year"]
            <= year_range[1]
        ),
        1,
        0
    )

    match_score += np.where(
        (
            movies["tomatometer_rating"]
            >= selected_tomatometer
        ),
        1,
        0
    )

    if match_score.max() > 0:
        match_norm = (
            match_score
            / match_score.max()
        )
    else:
        match_norm = match_score

    sentiment_norm = (
        sentiment_probs
        / sentiment_probs.max()
    )

    idx = movies[
        movies["movie_title"]
        == selected_movie
    ].index[0]

    cosine_sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    final_score = (
        alpha * cosine_sim
        + beta * match_norm
        + gamma * sentiment_norm
    )

    indices = final_score.argsort()[::-1]
    indices = [
        i
        for i in indices
        if i != idx
    ][:top_n]

    resultado = movies.iloc[
        indices
    ].copy()

    resultado[
        "sentiment"
    ] = sentiment_norm[
        indices
    ]

    return resultado


# ==========================================
# BOTON
# ==========================================

if st.button(
    "Obtener Recomendaciones"
):

    with st.spinner(
        "Generando..."
    ):

        recommendations = recomendar()

        for _, movie in (
            recommendations
            .iterrows()
        ):

            poster = obtener_poster(
                movie["movie_title"]
            )

            col1, col2 = st.columns(
                [1, 3]
            )

            with col1:

                if poster:
                    st.image(
                        poster,
                        use_container_width=True
                    )

            with col2:

                st.subheader(
                    movie["movie_title"]
                )

                st.write(
                    f"Tomatometer: "
                    f"{movie['tomatometer_rating']}%"
                )

                st.write(
                    f"Sentimiento: "
                    f"{movie['sentiment']:.2f}"
                )

                st.write(
                    movie["movie_info"]
                )

            st.divider()