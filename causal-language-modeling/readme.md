### Fine tuning Hugging Face for causal language modeling 
[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/layer/causal-language-modeling)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/causal-language-modeling/causal-language-modeling.ipynb)
[![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/causal-language-modeling)

In this project we fine tune a Hugging Face model for text generation on the wikitext dataset.
#### How to use
Make sure you have the latest version of Layer:

``` !pip install layer -q ``` 

Then, you can fetch the fine tuned model and the tokenizer from Layer and start generating text

```python
import layer
gpt2 = layer.get_model('layer/causal-language-modeling/models/gpt2-clm').get_train()
tokenizer = layer.get_model('layer/causal-language-modeling/tokenizer').get_train()
input_sequence = "I love reading books"
# encode context the generation is conditioned on
input_ids = tokenizer.encode(input_sequence, return_tensors='tf')
output = gpt2.generate(input_ids)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[0], skip_special_tokens = True))
# > I love reading books, and the song was released in the song. 
```
### Dataset
The model is fine tuned on the WikiText dataset. The WikiText is a collection of over 
100M token extracted from Wikipedia.  
## Model
We fine tune a GPT-2 model on the above dataset. 

GPT-2  is a model that has been pre-trained on a large corpus of English data. 
The data consists of web pages scrapped from the internet. 
The model can be fine-tuned for text classification. 

https://app.layer.ai/layer/causal-language-modeling/models/tokenizer  

https://app.layer.ai/layer/causal-language-modeling/models/gpt2-clm  