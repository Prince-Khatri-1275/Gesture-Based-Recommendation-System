import ast

import streamlit as st
import numpy as np
import pandas as pd

def get_popular_titles(reviews, df, min_ratings=50, k_populars=50):
    reviews_with_name_df = reviews.merge(df, left_on="id_2", right_on="id")[
        ["id_1", "title", "id_2", "score", "genre_encoded", "genre_score"]]
    num_ratings_df = reviews_with_name_df.groupby("title").count()["score"].reset_index()
    avg_score_df = (
                reviews_with_name_df.groupby("title").sum()["score"] / reviews_with_name_df.groupby("title").count()[
            "score"]).reset_index()

    num_ratings_df.rename(columns={"score": "num_ratings"}, inplace=True)
    avg_score_df.rename(columns={"score": "avg_score"}, inplace=True)

    popularity_df = avg_score_df.merge(num_ratings_df, on="title")
    popularity_df = popularity_df[popularity_df["num_ratings"] >= min_ratings]
    popularity_df = popularity_df.merge(df, on="title")[["num_ratings", "avg_score", "title", "genre_score"]]

    popularity_df["total_score"] = (popularity_df["avg_score"]/10+popularity_df["genre_score"])/2

    popularity_df = popularity_df.sort_values("total_score", ascending=False)[:k_populars]
    return popularity_df.merge(df, on="title")

# Defining required functions/methods
def get_genres(data): # Retrieve unique genres from given data
    genres = []
    for glist in data.genre:
        for gen in glist:
            gen = gen.lower()
            if gen not in genres:
                genres.append(gen)
    return sorted(genres)

def get_multi_genre_wise_indices(df, genres):
    genre_dict = {word:index for index, word in enumerate(sorted(get_genres(df)))}
    vectorized_genres = np.array(df.genre_encoded.tolist())
    n_genres = len(genre_dict)
    # st.write(vectorized_genres.shape, genres) # Debugging
    g_indices = [genre_dict[genre.lower()] for genre in genres]
    encoded_genre = np.sum(np.array([np.eye(n_genres)[g_index] for g_index in g_indices]), axis=0, keepdims=True).T

    # st.write(encoded_genre.shape)

    # Thought Feature
    # if threshold == -1:
    #     threshold = len(genres)
    #
    # threshold = min(len(genres), threshold)

    return vectorized_genres@encoded_genre / len(g_indices)

st.set_page_config(
    page_title="Multi-Genre-wise"
)

def fetch_anime(dfs, reviews, genres, k_recommendations=50, *, min_rating=50):
    dfs['genre_score'] = get_multi_genre_wise_indices(dfs, genres)
    # st.write(dfs)
    dataset = get_popular_titles(reviews, dfs, k_populars=k_recommendations, min_ratings=min_rating)
    start = 0
    left_entries = len(dataset)
    for i in range(-(-k_recommendations//5)):
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
                genre = desired_data["genre"]

                st.markdown(
                    f"""<a href="{anime_link}" target="_blank">
                                        <img src="{img_url}" width="150">
                                    </a>""",
                    unsafe_allow_html=True
                )
                st.markdown(f"{title}", help=f"Genre:-{genre}\n__________________\nSynopsis:-\n {synopsis}")
        start += 5


with st.expander("File Uploads"):
    # Reading main and ratings data
    main_data = st.file_uploader("Upload Main dataset", "csv")
    ratings = st.file_uploader("Upload ratings dataset", "csv")

if ratings and main_data:
    main_data = pd.read_csv(main_data)[["id", "title", "genre_encoded", "synopsis", "genre", "img_url", "link"]]
    ratings = pd.read_csv(ratings)[["id_1", "score", "id_2"]] # id_1 == User's id and id_2 == Data/Anime/Movie's id

    main_data.genre_encoded = main_data.genre_encoded.apply(ast.literal_eval)
    main_data.genre = main_data.genre.apply(ast.literal_eval)


    genre_list = get_genres(main_data)

    col1, col2, col3 = st.columns([2, 1, 1])
    curr_genres = col1.multiselect("Select_Genres", genre_list, default=genre_list[0])
    k_recommends = col2.slider("K_Recommendations", min_value=10, max_value=200, value=50, step=5)
    min_ratings = col3.slider("Minimum Ratings", min_value=10, max_value=200, value=50, step=5)

    # st.markdown(f"# {curr_genres}")
    if curr_genres:
        fetch_anime(main_data, ratings, curr_genres, k_recommendations=k_recommends, min_rating=min_ratings)