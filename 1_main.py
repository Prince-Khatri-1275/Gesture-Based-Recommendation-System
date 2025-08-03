import streamlit as st
import pandas as pd

def get_popular_titles(reviews, df, min_ratings=50, k_populars=50):
    reviews_with_name_df = reviews.merge(df, left_on="id_2", right_on="id")[
        ["id_1", "title", "id_2", "score"]]
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

st.set_page_config(
    page_title="Popular Anime"
)

def fetch_anime(dfs, reviews, k_recommendations=50, *, min_rating=50):
    dataset = get_popular_titles(reviews, dfs, k_populars=k_recommendations, min_ratings=min_rating)
    start = 0
    left_entries = len(dataset)
    for i in range((-(-k_recommendations//5))):
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

                st.markdown(
                    f"""<a href="{anime_link}" target="_blank">
                                        <img src="{img_url}" width="150">
                                    </a>""",
                    unsafe_allow_html=True
                )
                if "genre" in desired_data:
                    genre = desired_data["genre"]
                    st.markdown(f"{title}", help=f"Genre:-{genre}\n__________________\nSynopsis:-\n {synopsis}")
                else:
                    st.markdown(f"{title}", help=f"Synopsis:-\n {synopsis}")
        start += 5


with st.expander("File Uploads"):
    # Reading main and ratings data
    main_data = st.file_uploader("Upload Main dataset", "csv")
    ratings = st.file_uploader("Upload ratings dataset", "csv")
    col1, col2 = st.columns(2)
    min_ratings = col1.slider("Minimum Rating", min_value=0, max_value=200, value=50, step=5)
    k_recommends = col2.slider("K Recommendations", min_value=10, max_value=200, value=50, step=5)

if ratings and main_data:
    main_data = pd.read_csv(main_data)[["id", "title", "synopsis", "img_url", "link"]]
    ratings = pd.read_csv(ratings)[["id_1", "score", "id_2"]] # id_1 == User's id and id_2 == Data/Anime/Movie's id

    # st.markdown(f"# {curr_genres}")
    fetch_anime(main_data, ratings, k_recommendations = k_recommends, min_rating=min_ratings)