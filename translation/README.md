# Fine Tuning T5 for English to SQL Translation

[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/layer/t5-fine-tuning-with-layer) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/translation/T5_Fine_tuning_with_Layer.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/translation) [![Open Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blueviolet)](https://huggingface.co/spaces/mecevit/english-to-sql) 

In this project, we are going to fine tune [T5 by Google](https://github.com/google-research/text-to-text-transfer-transformer) with our custom data so that it can convert basic natural language questions to SQL queries.

## How to use

You can load and use the fine tuned model from this project easily. First make sure you have the latest version of Layer:
```
!pip install layer -q
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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q6wiwdyjPGbfABpsNwOBzhhWE5pQTdrC?usp=sharing) 

## Dataset

Unlike other language to language translation datasets, we can create our own English-SQL pairs easily with templates.

A sample template for dataset:

```
[prop1] of [nns] by [breakdown]
SELECT [prop1] , [breakdown] FROM [nns] GROUP BY [breakdown]
```

Here we can easily pass domain specific keys for the properties (prop1) or nouns (nns). You can preview the dataset we have created here:

https://app.layer.ai/layer/t5-fine-tuning-with-layer/datasets/english_sql_translations

## Model

![T5 Model](https://camo.githubusercontent.com/623b4dea0b653f2ad3f36c71ebfe749a677ac0a1/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f343030362f312a44304a31674e51663876727255704b657944387750412e706e67)
*Source: https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html*

The [T5 library](https://github.com/google-research/text-to-text-transfer-transformer) serves primarily as code for reproducing the experiments in Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. In the paper, we demonstrate how to achieve state-of-the-art results on multiple NLP tasks using a text-to-text transformer pre-trained on a large text corpus.

We are going to be using pretrained T5 model from :hugs: and fine tune it with adding a new task: `translate English to SQL`. We have used Layer GPU fabric to train the model for 3 Epochs. Here is the loss curve:

https://app.layer.ai/layer/t5-fine-tuning-with-layer/models/t5-english-to-sql?v=4.1&w=3.1&w=2.1#loss


You can find the models below with parameters and metrics:

https://app.layer.ai/layer/t5-fine-tuning-with-layer/models/t5-english-to-sql
https://app.layer.ai/layer/t5-fine-tuning-with-layer/models/t5-tokenizer

## References
- https://huggingface.co/docs/transformers/model_doc/t5
- https://github.com/google-research/text-to-text-transfer-transformer

## Citation Information

```
@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}
```