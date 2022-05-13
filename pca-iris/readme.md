# Principal Component Analysis using the Iris Dataset

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/douglas_mcilwraith/iris-pca/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/pca-iris/iris-pca.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/pca_iris)

We use the iris dataset to perform Principal Component Analysis. 

## Dataset

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

[source](https://en.wikipedia.org/wiki/Iris_flower_data_set)

## Technique

PCA is used in exploratory data analysis and for making predictive models. It is commonly used for dimensionality reduction by projecting each data point onto only the first few principal components to obtain lower-dimensional data while preserving as much of the data's variation as possible. The first principal component can equivalently be defined as a direction that maximizes the variance of the projected data. The i-th principal component can be taken as a direction orthogonal to the first i-1 principal components that maximizes the variance of the projected data.


[source](https://en.wikipedia.org/wiki/Principal_component_analysis)

## How to use 

Ensure you have the latest version of Layer: 
```
pip install layer --upgrade
```

This project will create a private 'iris-pca' project against which the dataset and model is stored 

## Model

PCA is performed and the first two components are used to plot the data in two dimensions (down from the original four). We make use of "layer.log()" to plot the resultant graph under the associated resources for this model.

```
