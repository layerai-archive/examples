# Train sentiment analysis model with Layer

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://development.layer.co/layer/sentiment-analysis) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/sentiment-analysis/sentiment_analysis.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/sentiment-analysis)

In this project we train sentiment analysis model using Recurrent Neural Networks in TensorFlow. 
## How to use

Make sure you have the latest version of Layer-SDK
```
!pip install layer-sdk -q
```

You can fetch the trained model and start making predictions from it right away. 
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array
import numpy as np
import layer
from sklearn.model_selection import train_test_split
review = "That was such a horrible movie, I hated it."
tokenizer = layer.get_model('layer/sentiment-analysis/models/imdb_data_tokenizer').get_train()
model = layer.get_model('layer/sentiment-analysis/models/tensorflow-sentiment-analysis')
classifier = model.get_train()
word_index = tokenizer.word_index
X_test_sequences = tokenizer.texts_to_sequences(review)
padding_type = "post"
truncation_type="post"
max_length = 512
X_test_padded = pad_sequences(X_test_sequences,maxlen=max_length, padding=padding_type,
                          truncating=truncation_type)
test_data = np.expand_dims(X_test_padded[0], axis=0)
prediction = classifier.predict(test_data)
if prediction[0][0]>0.5:
  print("Is positive")
else:
   print("Is a negative")
# > Is negative
```
## Dataset
In this project, we use the famous IMDB dataset to train a deep learning sentiment analysis model. 
The dataset can be seen using the link below: 
https://development.layer.co/layer/sentiment-analysis/datasets/imdb-dataset-of-50k-movie-reviews

You can start using this dataset immediately using the following command: 
```python
import layer
dataset = layer.get_dataset('layer/sentiment-analysis/datasets/imdb-dataset-of-50k-movie-reviews').to_pandas()


```
## Model
The model trained in this project is a Recurrent Neural Network. A Recurrent Neural Network is a special category of neural
networks that allows information to flow in both directions. An RNN has short-term memory that enables it to factor previous 
input when producing output. The short-term memory allows the network to retain past information and, hence, uncover
relationships between data points that are far from each other. RNNs are great for handling time series and sequence data such as audio and text.

Here is the model definition: 
```python
model = Sequential([
Embedding(vocab_size, 64, input_length=max_length),
Bidirectional(LSTM(64, return_sequences=True)),
Bidirectional(LSTM(64,)),
Dense(32, activation='relu'),
Dense(1, activation='sigmoid')])
```
The network consists of the following major building blocks:

- An **Embedding Layer**. A word embedding is a representation of words in a dense vector space. 
  In that space words that are semantically similar appear together. For instance, this can help in sentiment 
  classification in that negative words can be bundled together. The Keras embedding Layer expects us to pass the size of the vocabulary, the size of the dense embedding, and the length 
of the input sequences. This layer also loads pre-trained word embedding weights in transfer learning.


- **Bidirectional LSTMs** allow data to pass from both sides, that is from left to right and from right to left. 
  The output from both directions is concatenated but there is the option to sum, average, or multiply. 
  Bidirectional LSTMs help the network to learn the relationship between past and future words.  When two LSTMs are defined as shown below, the first one has to return sequences that will be passed to the next LSTM.

Follow the link below to see the models. 
  
https://development.layer.co/layer/sentiment-analysis/models/tensorflow-sentiment-analysis
https://development.layer.co/layer/sentiment-analysis/models/imdb_data_tokenizer