# Fine Tuning T5 for English to SQL Translation

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://development.layer.co/layer/t5-fine-tuning-with-layer) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QNEGlRKTnKn9VhS6vBTU03UFLx0ZgUao?usp=sharing) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/translation)

## How to use

Make sure you have the latest version of Layer-SDK
```
!pip install layer-sdk -q
```

Then, you can fetch the finetuned model and the tokenizer from Layer and start translating

```python
import layer

query = "Show me the average price of wines in Italy by provinces"

model = layer.get_model('layer/t5-fine-tuning-with-layer/models/t5-english-to-sql').get_train()
tokenizer = layer.get_model('layer/t5-fine-tuning-with-layer/models/t5-tokenizer').get_train()
input_ids = tokenizer.encode(f"translate English to SQL: {query}",return_tensors="pt")
outputs = model.generate(input_ids, max_length=1024)
tokenizer.decode(outputs[0],skip_special_tokens=True)

# > SELECT avg(price) , provinces FROM wines WHERE location = 'Italy' GROUP BY provinces
```

## Dataset

Unlike other language to language translation datasets, we can create our own English-SQL pairs easily with templates.

A sample template for dataset:

```
[prop1] of [nns] by [breakdown]
SELECT [prop1] , [breakdown] FROM [nns] GROUP BY [breakdown]
```

Here we can easily pass domain specific keys for the properties (prop1) or nouns (nns). You can preview the dataset we have created here:

https://development.layer.co/layer/t5-fine-tuning-with-layer/datasets/english_sql_translations

## Model

![T5 Model](https://camo.githubusercontent.com/623b4dea0b653f2ad3f36c71ebfe749a677ac0a1/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f343030362f312a44304a31674e51663876727255704b657944387750412e706e67)

The t5 library serves primarily as code for reproducing the experiments in Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. In the paper, we demonstrate how to achieve state-of-the-art results on multiple NLP tasks using a text-to-text transformer pre-trained on a large text corpus.

We are going to be using pretrained t5 model from :hugs: and fine tune it with adding a new task: `translate English to SQL`. Here is the versions of the model we have finetuned:

https://development.layer.co/layer/t5-fine-tuning-with-layer/models/t5-english-to-sql

