import ast
import os
import subprocess
from streamlit_extras.grid import grid
import streamlit as st
import numpy as np
import pandas as pd
import signal

st.set_page_config(
    page_title="Genre-wise"
)

PID_FILE = os.path.join("gesture_recognition", "camera_pid.txt")  # To track the process
script_path = os.path.join("gesture_recognition", "main.py")

# st.write(PID_FILE)
# st.write(script_path)

# Defining required functions/methods
def get_popular_titles(reviews, df, min_ratings=50, k_populars=50):
    reviews_with_name_df = reviews.merge(df, left_on="id_2", right_on="id")[
        ["id_1", "title", "id_2", "score", "genre_encoded"]]
    num_ratings_df = reviews_with_name_df.groupby("title").count()["score"].reset_index()
    avg_score_df = (
                reviews_with_name_df.groupby("title").sum()["score"] / reviews_with_name_df.groupby("title").count()[
            "score"]).reset_index()

    num_ratings_df.rename(columns={"score": "num_ratings"}, inplace=True)
    avg_score_df.rename(columns={"score": "avg_score"}, inplace=True)

    popularity_df = avg_score_df.merge(num_ratings_df, on="title")
    popularity_df = popularity_df[popularity_df["num_ratings"] >= min_ratings]

    popularity_df = popularity_df.sort_values("avg_score", ascending=False)[:k_populars]
    return popularity_df.merge(df, on="title")

def get_genres(data): # Retrieve unique genres from given data
    genres = []
    for glist in data.genre:
        for gen in glist:
            gen = gen.lower()
            if gen not in genres:
                genres.append(gen)
    return sorted(genres)

def get_genre_wise_indices(df, genre, threshold=0.5):
    genre_dict = {word:index for index, word in enumerate(sorted(get_genres(df)))}
    vectorized_genres = np.array(df.genre_encoded.tolist())
    n_genres = len(genre_dict)
    # st.write(vectorized_genres.shape, genre) # Debugging
    g_index = genre_dict[genre.lower()]
    encoded_genre = np.eye(n_genres)[g_index]

    return (vectorized_genres@encoded_genre)>=threshold

def fetch_anime(dfs, reviews, genre, k_recommends=50, *, min_rating=50):
    df = dfs[get_genre_wise_indices(dfs, genre, threshold=0.5)]
    dataset = get_popular_titles(reviews, df, k_populars=k_recommends, min_ratings=min_rating)
    start = 0
    left_entries = len(dataset)
    for i in range(10):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            left_entries -= 1
            if left_entries<0:
                break
            with col:
                desired_data = dataset.iloc[start + j]
                img_url = desired_data["img_url"]
                title = desired_data["title"]
                synopsis = desired_data["synopsis"]
                anime_link = desired_data["link"]
                genres = desired_data["genre"]

                st.markdown(
                    f"""<a href="{anime_link}" target="_blank">
                                        <img src="{img_url}" width="150">
                                    </a>""",
                    unsafe_allow_html=True
                )
                st.markdown(f"{title}", help=f"Genre:-{genres}\n__________________________\nSynopsis:-\n{synopsis}")
        start += 5

def get_genre_idx(swipe_to, genres, genre_idx):
    """

    :param swipe_to: takes value in between -1{left} and 1{right}
    :param genres: the genre list of the dataframe which is used for len mainly
    :param genre_idx: the current genre id
    :return: updated genre id
    """
    return (genre_idx+swipe_to)%len(genres)

# Reading main and ratings data
with st.expander("Load Files"):
    main_data = st.file_uploader("Upload Main dataset", "csv")
    ratings = st.file_uploader("Upload ratings dataset", "csv")

    script_cols = grid(2)
    if "run_script" not in st.session_state:
        st.session_state.run_script = False
    if "stop_script" not in st.session_state:
        st.session_state.stop_script = False

    if script_cols.button("Run Script", use_container_width=True):
        st.session_state.run_script = not st.session_state.run_script

    if script_cols.button("Stop Script", use_container_width=True):
        st.session_state.stop_script = not st.session_state.stop_script

if st.session_state.run_script:
    if not os.path.exists(PID_FILE):
        proc = subprocess.Popen(["python", script_path])
        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))
        # st.success("Camera started!")
    else:
        # st.warning("Camera is already running")
        pass
    st.session_state.run_script = False

if st.session_state.stop_script:
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            pid = int(f.read())
        try:
            os.kill(pid, signal.SIGTERM)
            os.remove(PID_FILE)
            # st.success("Camera stopped!")
        except Exception as e:
            st.error(f"Failed to stop camera: {e}")
    else:
        pass
        # st.warning("No running camera process to destroy.")
    st.session_state.stop_script = False

if ratings and main_data:
    main_data = pd.read_csv(main_data)[["id", "title", "genre_encoded", "synopsis", "genre", "img_url", "link"]]
    ratings = pd.read_csv(ratings)[["id_1", "score", "id_2"]] # id_1 == User's id and id_2 == Data/Anime/Movie's id

    main_data.genre_encoded = main_data.genre_encoded.apply(ast.literal_eval)
    main_data.genre = main_data.genre.apply(ast.literal_eval)

    if "genre_idx" not in st.session_state:
        st.session_state.genre_idx = 0

    grid_title = grid(1)
    grid_btn = grid([1, 1, 5, 5])

    move_left = grid_btn.button("<-")
    move_right = grid_btn.button("->")
    minimum_ratings = grid_btn.slider("Minimum rating", min_value=8, max_value=160, step=8, value=32)
    k_recommendations = grid_btn.slider("K Recommendations", min_value=8, max_value=160, step=8, value=32)
    genre_list = get_genres(main_data)

    if move_left:
        swiped = -1
        st.session_state.genre_idx = get_genre_idx(swiped, genre_list, st.session_state.genre_idx)
    if move_right:
        swiped = 1
        st.session_state.genre_idx = get_genre_idx(swiped, genre_list, st.session_state.genre_idx)

    # swiped = 0
    curr_genre = genre_list[st.session_state.genre_idx]

    grid_title.markdown(f"# {curr_genre.title()} ")

    fetch_anime(main_data, ratings, curr_genre, k_recommends=k_recommendations, min_rating=minimum_ratings)
