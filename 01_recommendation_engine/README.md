## Overview 
In these exercises, we will go over some of the techniques that we can use to build a recommender system. I chose to implement recommendation engine because it is still one of the most versatile ML application to be used in the production.

Why build a recommender system ?
- One of the oldest and most used Machine learning algorithm in production today. 
- Example use cases 
    - Item recommendations on a product page of an e-commerce websites, restaurant app
    - Hotels recommendations for "Similar hotels" 
    - Movie/song recommendations on online portals
    - Poems/jokes recommendations
    - Programming challenges recommendations
    - recipe recommendations give the user context (ingredients, food choices)
    - Similar questions on quora/stackoverflow.
    - Blog post recommendation on your blog site

- Any websites that has more than 10 products/services to sell/rent, can leverage recommendation models to suggest user experience.

Here we will look at two approaches 
- Content based: This approach leverages the static description of the items to be recommended. 
- Collaborative filtering/Matrix factorization: This approach utilizes user item interaction data to build recommendations. This interaction data can be implicit(clicks, views, other events) or explicit(ratings, feedback, order_value). For explicit and sparse interaction matrix, we use matrix factorisation. On the other hands, we can use collaborative filtering for implicit feedbacks.
- Hybrid models: This approach leverages both the data (content based and interaction based). It combines the output either by 
    - combining the candidate list from both the sources 
    - using one as a fallback 
    - uses content-based recommendation for new users 


## Running the test 
on command line run 
```bash 
python -m unittest test_content.py
```
Currently, it will show that all tests are being skipped. But, as you implement the steps, you should run the tests after removing the skip decorator.

## Content based recommendation Engine 
Content based recommendation engines (RE) rely on static information provided for the item. Here is what it means for some of the items 
- Product: name, description, category, price 
- Recipe: ingredient, allergens, steps involved, cuisine 
- Quora question: question, topic, tags 
- Hotel: Locality, city, amenities, star ratings, price, distance from airport/stations, distance from tourist destinations, 
- Movies: Cast, title, director, genre, awards nominations, language, theme
- Programming challenge(e.g. Leetcode): data structure used, difficulty level, name of companies where it was asked, programming concepts

These are some of the static attributes associated with an item. These rarely change. Given these attrribute, we will build a content similarity recommendation engine. There are many ways to build an algorithm to do it. In these exercise we will go through the following steps to understand the REs more. 
- Step 0: Setup the environment and get familiar with the data. Get top 5 geners for animes in the animes dataset 
- Step 1: REs can be broken down into two steps candidate generation and ranking. Here we will implement a simple ranking algorithm which takes first 50 entries containing most members and then sort them by ratings. Choice of fields to sort by is arbitrary and generally requires an understanding of the domain. 
- Step 2: Here we will implement simple strategy of generating candidate which share the same genre with the given anime.
- Step 3: Here we will implement the strategy of generating candidate set which share keywords in the name with teh given anime. Since our dataset is small, we can load the whole data in memory and generate recommendations on fly. But for larger datasets, we will need to pre-process the data before and use the artifacts generated to serve recommednations on the fly. 
- Step 4: Here we will run a Nearest Neighbor model to find the candidate list. For measuring distance between two animes, we will use jaccard similarity metric between genres corresponding to the animes. 

## Matrix factorization (SVD) method

Step 1. Here we will implement SVD model for factorizing the user-item feedback matrix. [Mathematical foundation](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) behind the SVD model involves stochastic gradient descent to compute biases terms for each user and items and latent vector for each user and item. 


