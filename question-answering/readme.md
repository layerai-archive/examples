### Question answering with SQUAD dataset and Transformers
[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/layer/qas/models/qa)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1h8M_sLKAbvsAA11qgSPsOc98g08RB4_u/view?usp=sharing)
[![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples)

In this project we fine tune a Hugging Face BERT model on the SQUAD dataset for question answering.


#### How to use
Make sure you have the latest version of Layer-SDK

``` !pip install layer-sdk -q ``` 

Then, you can fetch the fine tuned model and the tokenizer from Layer and start generating text

```python
import tensorflow as tf
import layer
context = """
On a mild day in late July, a teenager was walking home from the downtown
area, such as it was, of Little Whinging. The city did not have a train station, a
department store, or even a large supermarket; just a few dozen shops, a bank, a
police station, and a library. It was the library from which soon-to-be-sixteen-yearold Harry Potter was returning, with an old bookbag around his shoulder. He
looked around from time to time as he walked. It looked as though he were
admiring the trees and bushes, which had recovered nicely from last year’s drought,
but he actually was wondering whether there was anyone following him. Or, more
precisely, whether he could catch a glimpse of the person he knew must be
following him. All he could see, however, were the normal sights of a suburban
neighborhood, and a few people looking at him rather oddly as they passed him.
Harry briefly wondered why–after all, he was not exactly famous in this area, nor
was his scar–until he realized that looking around to see if you were being followed
was not exactly usual behavior. 
"""
tokenizer = layer.get_model("layer/qas/models/dsbt-tokenizer").get_train()

question = "When was the teenager walking home?"
model = layer.get_model("layer/qas/models/qa").get_train()
question = "When was the teenager walking home?"
inputs = tokenizer([context], [question], return_tensors="np")
outputs = model(inputs)
start_position = tf.argmax(outputs.start_logits, axis=1)
end_position = tf.argmax(outputs.end_logits, axis=1)
print(int(start_position), int(end_position[0]))
answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
print(answer)
print(tokenizer.decode(answer))
# > 0 7
# [  101  2006  1037 10256  2154  1999  2397  2251]
# [CLS] on a mild day in late july
```
### Dataset 
In this project we use the [Stanford Question Answering Dataset (SQuAD) dataset](https://huggingface.co/datasets/squad). 
The dataset is made up of some questions and answers. 
### Model 
We use the SQuAD dataset to fine tune a [DistilBert](https://huggingface.co/docs/transformers/model_doc/distilbert
) for the question and answering task. 

The models saved as a result of this work can be seen below: 

https://app.layer.ai/layer/qas/models/qa

https://app.layer.ai/layer/qas/models/dsbt-tokenizer
