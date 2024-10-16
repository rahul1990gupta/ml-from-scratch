"""
Outline of the project 
Rule based approach for content-based recommendation engine

Ranking 
- Top 50 items by members count 
- of those 50 items, select top 10 items by rating

Candidate list generation
- Suggest  items in the same genre 
- Treat anime name as bag of words after removing stopwords find titles with 
some intersection.
"""

import pandas as pd 
from functools import reduce
from collections import Counter
from operator import itemgetter 
import numpy as np


func = lambda x, y: x + y

with open("./stopwords.txt", "r") as f:
    stopwords = [line.strip() for line in f.readlines()]

anime_df = pd.read_csv(
    "anime/anime_info.dat", 
    sep="\t"
)

anime_df.set_index("anime_ids", inplace=True)


genre_ll = anime_df.genre.dropna().str.split(",").apply(lambda x : list(map(str.strip, x))).tolist()

genre_list = reduce(func, genre_ll, [])

genres = sorted(set(genre_list))

gtoi = {g: i for i, g in enumerate(genres)} 
itog = {i: g for i, g in enumerate(genres)}


def top_genre(n=5):
    anime_ll = anime_df.genre.dropna().str.split(",").tolist()
    anime_list = reduce(func, anime_ll, [])

    anime_set = Counter(anime_list)
    return [element[0] for element in anime_set.most_common(n)]

def count_words_in_name():
    word_ll = anime_df.name.dropna().str.split(" ").tolist()
    word_list = reduce(func, word_ll, [])
    c = Counter(map(str.lower, word_list))
    return c

def candidate_list_in_same_genre(anime_id):
    genres = anime_df.loc[anime_id].genre
    
    dfs = []
    for genre in genres.split(","):
        genre = genre.strip()  # Remove any leading/trailing spaces
        mask = anime_df['genre'].apply(lambda x: genre in x if isinstance(x, str) else False)
        
        df = anime_df[mask]
        dfs.append(df)
    df = pd.concat(dfs)
    candidate_ids = set(df.index)
    return candidate_ids


def candidate_list_with_matching_names_kw(anime_id):
    name = anime_df.loc[anime_id]["name"]

    dfs = []
    # implement stopwords
    for token in name.split(" "):
        if token in stopwords:
            continue

        mask = anime_df["name"].apply(lambda x: token in x if isinstance(x, str) else False)
        df = anime_df[mask]
        dfs.append(df)
    
    df = pd.concat(dfs)
    candidate_ids = set(df.index)
    return candidate_ids


# similarity in sets
def jaccard(x, y):
    intersection = set(x).intersection(set(y))
    union = set(x).union(set(y))

    score = len(intersection)/len(union)
    
    return score 


def candidate_list_with_knn(anime_id):
    genres = anime_df.loc[anime_id].genre.split(",")
    key_genres = [g.strip() for g in genres]

    data = anime_df[["genre"]].to_records()
    
    scores = []
    for id, genre_str in data:
        if genre_str is np.nan or id == anime_id:
            continue
        
        score = (
            id, 
            jaccard(
                key_genres, 
                [g.strip() for g in genre_str.split(",")]
            )
        ) 

        scores.append(score)

    results = sorted(scores, key=itemgetter(1), reverse=True)

    return [r[0] for r in results if r[1]> 0.7]


def rank_by_member_rating(anime_ids):
    candidate_df = anime_df.loc[anime_ids]

    top_50 = candidate_df.sort_values("members", ascending=False).head(50)
    result = top_50.sort_values("rating", ascending=False)
    return result.index.tolist()


def run(command, anime_ids=[]):
    if command == "genre":
        return top_genre()
    if command == "rank":
        return rank_by_member_rating(anime_ids)
    if command == "candidate_same_genre":
        return candidate_list_in_same_genre(anime_ids[0])
    if command == "candidate_matching_names":
        return candidate_list_with_matching_names_kw(anime_ids[0])
    if command == "candidate_knn":
        return candidate_list_with_knn(anime_ids[0])

