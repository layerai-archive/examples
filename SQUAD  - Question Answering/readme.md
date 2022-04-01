### Question answering with SQUAD dataset and Transformers

#### How to use
Make sure you have the latest version of Layer-SDK

``` !pip install layer-sdk -q ``` 

Then, you can fetch the finetuned model and the tokenizer from Layer and start generating text

```
model = layer.get_model('qa').get_train()
context = """Keras is an API designed for human beings, not machines. Keras follows best
practices for reducing cognitive load: it  offers consistent & simple APIs, it minimizes
the number of user actions required for common use cases, and it provides clear &
actionable error messages. It also has extensive documentation and developer guides. """
question = "What is Keras?"

inputs = tokenizer([context], [question], return_tensors="np")
outputs = model(inputs)
start_position = tf.argmax(outputs.start_logits, axis=1)
end_position = tf.argmax(outputs.end_logits, axis=1)
print(int(start_position), int(end_position[0]))
answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
print(answer)
print(tokenizer.decode(answer))
```

https://development.layer.co/layer/qas/models/qa