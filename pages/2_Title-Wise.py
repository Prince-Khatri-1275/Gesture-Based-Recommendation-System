import ast

import pandas as pd
import pickle
import streamlit as st
from streamlit_extras import grid

st.title("My Recommender System")

with st.expander("Step 1: Upload Files"):
    main_data = st.file_uploader("Upload data file", "csv")
    # vocab_ = st.file_uploader("Upload pickled vocabulary", "pkl")
    sparse_score = st.file_uploader("Upload sparse score", "pkl")

is_all_uploaded = bool(main_data and sparse_score)
if is_all_uploaded:
    main_data = pd.read_csv(main_data)
    # vocab_ = pickle.load(vocab_)
    sparse_score = pickle.load(sparse_score)

def recommend(df, title=None, k=10):
    index = df[df["title"]==title].index
    index = index[0]
    query = sparse_score.getrow(index) # or can use .T transposed matrix
    query_dict = {i: val for i, val in enumerate(query.toarray()[0])}
    similar_recommendations = sorted(query_dict, key = lambda x: query_dict[x], reverse=True)

    del query, query_dict
    
    return df.loc[similar_recommendations][:k+1]

def show_posters(df, k_recommends=10, i_cols=5):
    n = -(-k_recommends//i_cols)
    for i in range(n):
        cols = st.columns(i_cols)
        for j, col in enumerate(cols):
            with col:
                desired_data = df.iloc[i*i_cols  +j]
                img_url = desired_data["img_url"]
                title = desired_data["title"]
                synopsis = desired_data["synopsis"]
                link = desired_data["link"]
                genre = ", ".join((desired_data["genre"]))

                st.markdown(
                    f"""<a href="{link}" target="_blank">
                                        <img src="{img_url}" width="150">
                                    </a>""",
                    unsafe_allow_html=True
                )
                st.markdown(f"{title}", help=f"Genre:-{genre}\n_________________\nSynopsis:-\n {synopsis}")
                k_recommends -= 1
                if k_recommends<=0:
                    break

if is_all_uploaded:
    with st.expander("Step 2:- Recommendations"):
        rec_preq_grid = grid.grid(2)

        selected_title = rec_preq_grid.selectbox("Select title for recommendations", main_data.title.values)
        k_recommendations = rec_preq_grid.slider("Select number of recommendations", min_value=5, max_value=50, value=10, step=1)

        main_data.genre = main_data.genre.apply(ast.literal_eval)

        if st.button("Recommend"):
            show_posters(recommend(main_data, title=selected_title, k=k_recommendations), k_recommendations+1, i_cols=5)
