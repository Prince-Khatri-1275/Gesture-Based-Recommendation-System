# Importing necessary libraries
import ast
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Parse"
)

main_data = st.file_uploader("Upload Main dataset", "csv")

# Defining required functions/methods
def get_genres(data): # Retrieve unique genres from given data
    genres = []
    for glist in data.genre:
        for gen in glist:
            gen = gen.lower()
            if gen not in genres:
                genres.append(gen)
    return sorted(genres)

def get_multi_hot_encoded_genre(genres, genre_list):
    genre_to_unk_code = {word:index for index, word in enumerate(sorted(genre_list))}
    n = len(genre_to_unk_code)
    vector = [0 for i in range(n)]
    for genre in genres:
        index = genre_to_unk_code[genre.lower()]
        vector[index] = 1

    return vector


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

def get_similarity_scores(synopsis:pd.Series, vectorized_genre:pd.Series, max_features=2500, min_threshold=0.25):
    # Importing necessary libraries
    import re  # Cleaning data
    import pickle  # Saving similarity and vocab files
    from nltk.stem.porter import PorterStemmer  # Removing similar meaning words and extracting only main ones
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix

    ps = PorterStemmer()
    cv = CountVectorizer(max_features=max_features, stop_words="english")

    # Creating necessary functions
    def clean(text):
        sentence = re.sub(r'[^\w\s]', '', str(text))  # Removes everything except words and spaces
        return sentence

    def stem(text):
        L = []
        for word in text.split():
            L.append(ps.stem(word.lower()))

        return " ".join(L)

    synopsis = synopsis.apply(clean).apply(stem)
    vectors = cv.fit_transform(synopsis)

    vectorized_genre = np.stack(vectorized_genre)
    del synopsis
    del stem, clean, CountVectorizer, PorterStemmer

    similarity_syn = cosine_similarity(vectors)*0.7
    del vectors
    similarity_gen = cosine_similarity(vectorized_genre)*0.3
    del vectorized_genre

    similarity_score = similarity_syn+similarity_gen
    del similarity_syn, similarity_gen

    similarity_score[similarity_score<min_threshold] = 0
    sparse_similarity = csr_matrix(similarity_score)

    del similarity_score
    return pickle.dumps(sparse_similarity), pickle.dumps(cv.vocabulary_)

def filter_data(df):
    df["genre"] = df["genre"].apply(ast.literal_eval)

    genre_list = get_genres(df)
    df['genre_encoded'] = df.genre.apply(get_multi_hot_encoded_genre, genre_list=genre_list)

    # To remove duplicates from the data and resetting the index after wards
    df.drop_duplicates(["title", "id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

if main_data is not None:
    main_data = pd.read_csv(main_data)[["id", "title", "genre", "synopsis", "img_url", "link"]]
    main_data = filter_data(main_data)

    if st.button("Show dataset!"):
        st.dataframe(main_data)

    st.download_button(
        "Download parsed file",
        data=main_data.to_csv(index=False).encode("utf-8"),
        file_name="main_parsed.csv",
        mime="csv/xlsx"
    )

    if "want_sim" not in st.session_state:
        st.session_state.want_sim = False

    if st.button("Want sparse similarity matrix"):
        st.session_state.want_sim = not st.session_state.want_sim

    if st.session_state.want_sim:
        with st.spinner("Calculating sparse similarity matrix..."):
            sparse_score, vocab_ = get_similarity_scores(
                main_data.synopsis,
                main_data.genre_encoded,
                max_features=3000,
                min_threshold=0.25
            )

            st.success("Calculation complete")

        col1, col2 = st.columns(2)
        col1.download_button(
            "Download sparse scores",
            data=sparse_score,
            file_name="sparse_score.pkl",
            mime="application/octet-stream"
        )

        col2.download_button(
            "Download vocabulary",
            data=vocab_,
            file_name="vocab_.pkl",
            mime="application/octet-stream"
        )
