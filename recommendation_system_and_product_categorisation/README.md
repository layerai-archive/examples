# E-commerce Recommendation Systems & Product Categorisation

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/Recommendation_System_and_Product_Categorisation_Project/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/recommendation_engine_and_product_categorisation/recommendation_system_and_product_categorisation/Recommendation_System_%26_Product_Categorisation.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/recommendation_engine_and_product_categorisation/recommendation_system_and_product_categorisation)

In this e-commerce example walkthrough, we will develop and build a Recommendation System on  Layer. We will use a public clickstream dataset for this example project. For more information about the dataset, you could check out its Kaggle page here: https://www.kaggle.com/datasets/tunguz/clickstream-data-for-online-shopping

## Methodology in a Nutshell

<img src="https://github.com/layerai/examples/blob/recommendation_engine_and_product_categorisation/recommendation_system_and_product_categorisation/methodology.png" height="60" width="60" >

For more information about this approach, we recommend you to read this comprehensive article: 
https://towardsdatascience.com/ad2vec-similar-listings-recommender-for-marketplaces-d98f7b6e8f03


## How to use

Make sure you have the latest version of Layer:
```
!pip install layer -q
```

```python
import layer

def get_5_recommendations(product_name):
  # import random 
  from random import sample
  df = layer.get_dataset("final_product_clusters").to_pandas()

  # Randomly select sample with 5 product ids
  five_recommendations = sample(df[df["Product_ID"]==product_name]["Cluster_Member_List"].tolist()[0].tolist(),5)
  # Exclude the given product 
  while product_name in five_recommendations:
    five_recommendations = sample(df[df["Product_ID"]==product_name]["Cluster_Member_List"].tolist()[0].tolist(),5)

  return five_recommendations  

  
get_5_recommendations("A13")
```
Result will look like: 

['B19', 'A10', 'C51', 'A6', 'C44']

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DU7GUaKJkSLDMTHus5b8nfBxG0rooPn2?usp=sharing) 

## Datasets

We will use the famous [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) to train our model. We will only use 3 of those datasets: *olist_orders_dataset.csv* & *olist_order_items_dataset.csv* & *olist_order_reviews_dataset.csv*. Please check descriptions of these datasets from its Kaggle link above.

We have created total of 9 Layer datasets in this project. Here is the list of those datasets and their little descriptions.

From the *olist_orders_dataset.csv* file, we have created 3 datasets:

*  **orders_raw_table:** This is basically identical to the csv file. It just Layer Dataset definition of the same orders raw data.

https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/datasets/orders_raw_table

* **orders_clean_table:** This is the clean version of the orders data after applying some data transformation operations on the orders_raw_table. 

https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/datasets/orders_clean_table

* **orders_based_features:** High level features extracted from the orders_clean_table.

https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/datasets/orders_based_features


From the *olist_order_items_dataset.csv* file, we have created 3 datasets:
* **items_raw_table:** This is basically identical to the csv file. It just Layer Dataset definition of the same items raw data.
 
https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/datasets/items_raw_table

* **items_clean_table:** This is the clean version of the items data after applying some data transformation operations on the items_raw_table. 

https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/datasets/items_clean_table

* **items_based_features:** High level features extracted from the items_clean_table.

https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/datasets/items_based_features


From the *olist_order_reviews_dataset.csv* file, we have created 2 datasets

* **reviews_raw_table:** This is basically identical to the csv file. It just Layer Dataset definition of the same reviews raw data.

https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/datasets/reviews_raw_table

* **reviews_clean_table:** This dataset is created to extract target variable for the problem which is the review scores for the past orders. 


Finally, we created the training_data which merges the orders_based_features, items_based_features and reviews_clean_table. This dataset is used to train the model.

* **training_data:**  
https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/datasets/training_data




## Model

We will be training a XGBRegressor from xgboost. We will fit the training dataset we have created. You can find all the model experiments and logged data here:

* **review_score_predictor_model:**
https://app.layer.ai/layer/ecommerce_olist_order_review_score_prediction/models/review_score_predictor_model

#### Acknowledgements
Thanks to [Olist](https://olist.com/pt-br/) for releasing this dataset.
