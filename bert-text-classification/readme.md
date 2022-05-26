# Sentiment Classification of IMDB Reviews Using DistilBERT

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/douglas_mcilwraith/bert-text-classification/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/bert-text-classification/distilbert-imdb.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/bert-text-classification)

In this example we use the [DistilBERT](https://arxiv.org/abs/1910.01108) language model, a smaller, faster a variant of the popular [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) model.

BERT provides contextualised language embeddings. That is to say, that the use of the same word in a different sentence will provide a different embedding. It is trained by minimising the loss of two tasks 1) the prediction of masked words in a sentence and 2) next sentence prediction. DistilBERT is a 'distilled' version of BERT which reduces the size of a BERT model by 40% through the use of a knowledge distillation process.

We use the pretrained [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) from HuggingFace. This model has been trained on the `wikipedia` and `bookcorpus` dataset. We then fine-tune this model on a sentiment classification task using a subset of the popular [`imdb`](https://huggingface.co/datasets/imdb) dataset. This dataset provides a total of 50000 (half train, half test) highly polar movie reviews and labelled their sentiment (positive vs negative). Testing is performed over 5 folds of the test set and the results are logged in Layer. 


## How To Use

First make sure you have the latest version of Layer:

```python
!pip install layer -q
```

Datasets can be uploaded to layer as follows. For this project, we use the the 50% split of train/test data to create two Layer datasets. `imdb-train` and `imdb-test`. The following code illustrates the process for one of these datasets.

```python
@dataset("imdb-train")
@pip_requirements(packages=["datasets"])
def build():
    from datasets import load_dataset
    import pandas as pd

    ds = load_dataset("imdb")['train']
    df = pd.DataFrame(ds)
    return df

layer.run([build])
```



## Dataset

The [Iris flower data set](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems" as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

Source: [Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)

## Model

[PCA](https://doi.org/10.1080/14786440109462720) was invented in 1901 by Karl Pearson, as an analogue of the principal axis theorem in mechanics; it was later independently developed and named by Harold Hotelling in the 1930s is used in exploratory data analysis and for making predictive models. 

It is commonly used for dimensionality reduction by projecting each data point onto only the first few principal components to obtain lower-dimensional data while preserving as much of the data's variation as possible. The first principal component can equivalently be defined as a direction that maximizes the variance of the projected data. The `i-th` principal component can be taken as a direction orthogonal to the first `i-1`principal components that maximizes the variance of the projected data.

Source: [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)

### References

- Devlin, Jacob, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” ArXiv abs/1810.04805 (2019)
- Sanh, Victor, Lysandre Debut, Julien Chaumond and Thomas Wolf. “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.” ArXiv abs/1910.01108 (2019):

