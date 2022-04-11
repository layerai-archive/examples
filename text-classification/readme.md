### Fine tuning Hugging Face for text classification
[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://development.layer.co/layer/derrick-bert)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/text-classification/text-classification-fine-tuning-hf.ipynb)
[![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/text-classification)
### How to use
Make sure you have the latest version of Layer-SDK

``` !pip install layer-sdk -q ``` 

Then, you can fetch the fine tuned model and the tokenizer from Layer and start classifying text.

```
import layer
bert = layer.get_model('bert').get_train()
tokenizer = layer.get_model('bert-tokenizer').get_train()
input_sequence = "I really loved that movie, the script was on point"
# encode context the generation is conditioned on
input_ids = tokenizer.encode(input_sequence, return_tensors='tf')
output = bert(input_ids)
logits = output.logits
import tensorflow as tf
predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
bert.config.id2label[predicted_class_id]
# > LABEL_1
```
### Dataset 
In this example, we use the famous [IMDB](derrick/HF-text-classification-fine-tuning) dataset to fine tuning a text classification model. 
The dataset has two labels; `0` and `1`. 
### Model 
We fine tune a pre-trained [BERT uncased model](https://huggingface.co/bert-base-uncased). It doesn't distinguish between English and 
english. BERT has been pre-trained on a large corpus of English data and can also be used for 
Masked Language Modeling and Next sentence Prediction. 

https://development.layer.co/layer/derrick-bert/models/bert-tokenizer  
https://development.layer.co/layer/derrick-bert/models/bert  