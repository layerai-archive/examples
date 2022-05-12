# E-commerce Recommendation Systems & Product Categorisation

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/Recommendation_System_and_Product_Categorisation_Project/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/recommendation-system/Recommendation_System_and_Product_Categorisation.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/recommendation-system)

In this e-commerce example walkthrough, we will develop and build a Recommendation System on  Layer. We will use a public clickstream dataset for this example project. For more information about the dataset, you could check out its Kaggle page here: https://www.kaggle.com/datasets/tunguz/clickstream-data-for-online-shopping

## Methodology in a Nutshell

![Methodology](https://github.com/layerai/examples/raw/main/recommendation-system/methodology_plot.png)


For more information about this approach, we recommend you to read this comprehensive article: 
https://towardsdatascience.com/ad2vec-similar-listings-recommender-for-marketplaces-d98f7b6e8f03


## How to use

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q4gCY44bSiwgMjkTyop6KTTJNtj-FxhT?usp=sharing) 

Make sure you have the latest version of Layer:
```
!pip install layer -q
```

```python
import layer

def get_5_recommendations(product_name):
  # import random 
  from random import sample
  df = layer.get_dataset("layer/Recommendation_System_and_Product_Categorisation_Project/datasets/final_product_clusters").to_pandas()

  # Randomly select sample with 5 product ids
  five_recommendations = sample(df[df["Product_ID"]==product_name]['Cluster_Member_List'].iloc[0].tolist(), 5)
  # Exclude the given product 
  while product_name in five_recommendations:
    five_recommendations = sample(df[df["Product_ID"]==product_name]['Cluster_Member_List'].iloc[0].tolist(), 5)

  return five_recommendations  

  
get_5_recommendations("A13")
```
Result will look like: 

['B19', 'A10', 'C51', 'A6', 'C44']

## Datasets

We have created total of 4 Layer datasets in this project. Here is the list of those datasets and their little descriptions.

*  **raw_session_based_clickstream_data:** This is basically identical to the source csv file which is a public Kaggle dataset: https://www.kaggle.com/datasets/tunguz/clickstream-data-for-online-shopping

    It is just Layer Dataset definition of the same clickstream raw data.

https://app.layer.ai/layer/Recommendation_System_and_Product_Categorisation_Project/datasets/raw_session_based_clickstream_data

* **sequential_products:** This is a Layer dataset derived from the previous dataset which consists of sequences of products viewed in order per session. 

https://app.layer.ai/layer/Recommendation_System_and_Product_Categorisation_Project/datasets/sequential_products

* **product_ids_and_vectors:** This is a Layer dataset which stores product vectors (embeddings) returned from Word2Vec algorithm.

https://app.layer.ai/layer/Recommendation_System_and_Product_Categorisation_Project/datasets/product_ids_and_vectors

* **final_product_clusters:** This is a Layer dataset which stores assigned cluster numbers per product and other members of those clusters.
 
https://app.layer.ai/layer/Recommendation_System_and_Product_Categorisation_Project/datasets/final_product_clusters


## Model

We will be training a K-Means model from sklearn. We will fit our clustering model using product vectors that we have created previously. You can find all the model experiments and logged data here:

* **clustering_model:**
https://app.layer.ai/layer/Recommendation_System_and_Product_Categorisation_Project/models/clustering_model

#### Acknowledgements
Thanks to [ÅapczyÅ„ski M., BiaÅ‚owÄ…s S. (2013) Discovering Patterns of Users' Behaviour in an E-shop - Comparison of Consumer Buying Behaviours in Poland and Other European Countries, â€œStudia Ekonomiczneâ€, nr 151, â€œLa sociÃ©tÃ© de l'information : perspective europÃ©enne et globale : les usages et les risques d'Internet pour les citoyens et les consommateursâ€, p. 144-153](https://olist.com/pt-br/) for releasing this dataset.
