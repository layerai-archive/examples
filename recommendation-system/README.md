# E-Commerce Recommendation System

[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/layer/Ecommerce_Recommendation_System/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/recommendation-system/Ecommerce_Recommendation_System.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/recommendation-system)

In this e-commerce example walkthrough, we will develop and build a Recommendation System on  Layer. We will use a public clickstream dataset for this example project. For more information about the dataset, you could check out its Kaggle page here: https://www.kaggle.com/datasets/tunguz/clickstream-data-for-online-shopping

## Methodology in a Nutshell

![Methodology](https://github.com/layerai/examples/raw/main/recommendation-system/methodology_plot.png)


For more information about this approach, we recommend you to read this comprehensive article: 
https://towardsdatascience.com/ad2vec-similar-listings-recommender-for-marketplaces-d98f7b6e8f03


## How to use

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/recommendation-system/How_to_use.ipynb) 

Make sure you have the latest version of Layer:
```
!pip install layer -q
```

```python
import layer
import numpy as np

# Fetch the K-Means model from Layer
kmeans_model = layer.get_model("layer/Ecommerce_Recommendation_System/models/clustering_model").get_train()

# Fetch product vectors (embeddings) dataset from Layer
product_ids_and_vectors = layer.get_dataset("layer/Ecommerce_Recommendation_System/datasets/product_ids_and_vectors").to_pandas()

# Product ID to generate recommendations for - You could try different product IDs in the data such as A16, C17, P12 etc.
product_id = "A13"

# Get Vector (Embedding) array of the given product
vector_array = np.array(product_ids_and_vectors[product_ids_and_vectors["Product_ID"]==product_id]["Vectors"].tolist())

# Get cluster number for the given product assigned by the model
cluster_no = kmeans_model.predict(vector_array)[0]

# Fetch final clusters members list dataset from Layer
final_product_clusters = layer.get_dataset("layer/Ecommerce_Recommendation_System/datasets/final_product_clusters").to_pandas()

# Get members list of the cluster that the given product is assigned to 
cluster_members_list = final_product_clusters[final_product_clusters['Cluster_No']==cluster_no]['Cluster_Member_List'].iloc[0].tolist()

# Randomly select 5 product recommendations from the cluster members excluding the given product
from random import sample
cluster_members_list.remove(product_id)
five_product_recommendations = sample(cluster_members_list, 5)

print("5 Similar Product Recommendations for {}: ".format(product_id),five_product_recommendations)
  
```
5 Similar Product Recommendations for A13:  ['C17', 'P60', 'C44', 'P56', 'A6']

## Datasets

We have created total of 4 Layer datasets in this project. Here is the list of those datasets and their little descriptions.

*  **raw_session_based_clickstream_data:** This is basically identical to the source csv file which is a public Kaggle dataset: https://www.kaggle.com/datasets/tunguz/clickstream-data-for-online-shopping

    It is just Layer Dataset definition of the same clickstream raw data.

https://app.layer.ai/layer/Ecommerce_Recommendation_System/datasets/raw_session_based_clickstream_data

* **sequential_products:** This is a Layer dataset derived from the previous dataset which consists of sequences of products viewed in order per session. 

https://app.layer.ai/layer/Ecommerce_Recommendation_System/datasets/sequential_products

* **product_ids_and_vectors:** This is a Layer dataset which stores product vectors (embeddings) returned from Word2Vec algorithm.

https://app.layer.ai/layer/Ecommerce_Recommendation_System/datasets/product_ids_and_vectors

* **final_product_clusters:** This is a Layer dataset which stores assigned cluster numbers per product and other members of those clusters.
 
https://app.layer.ai/layer/Ecommerce_Recommendation_System/datasets/final_product_clusters


## Model

We will be training a K-Means model from sklearn. We will fit our clustering model using product vectors that we have created previously. You can find all the model experiments and logged data here:

* **clustering_model:**
https://app.layer.ai/layer/Ecommerce_Recommendation_System/models/clustering_model

#### Acknowledgements
Thanks to [ÅapczyÅ„ski M., BiaÅ‚owÄ…s S. (2013) Discovering Patterns of Users' Behaviour in an E-shop - Comparison of Consumer Buying Behaviours in Poland and Other European Countries, â€œStudia Ekonomiczneâ€, nr 151, â€œLa sociÃ©tÃ© de l'information : perspective europÃ©enne et globale : les usages et les risques d'Internet pour les citoyens et les consommateursâ€, p. 144-153](https://olist.com/pt-br/) for releasing this dataset.
