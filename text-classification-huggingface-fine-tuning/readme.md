### Fine tuning Hugging Face for text classification

#### How to use
Make sure you have the latest version of Layer-SDK

``` !pip install layer-sdk -q ``` 

Then, you can fetch the finetuned model and the tokenizer from Layer and start generating text

```
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
```

https://development.layer.co/layer/derrick-bert/models/bert-tokenizer  
https://development.layer.co/layer/derrick-bert/models/bert  