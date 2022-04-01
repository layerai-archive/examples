### Text generation with GPT2

#### How to use
Make sure you have the latest version of Layer-SDK

``` !pip install layer-sdk -q ``` 

Then, you can fetch the finetuned model and the tokenizer from Layer and start generating text

```
gpt2 = layer.get_model('gpt-2').get_train()

tokenizer = layer.get_model('tokenizer-tester').get_train()
input_sequence = "The farmer planted a lot of crops"

# encode context the generation is conditioned on
input_ids = tokenizer.encode(input_sequence, return_tensors='tf')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = gpt2.generate(input_ids,max_length=100)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens = True))
```

https://development.layer.co/layer/sandbox-config/models/tokenizer-tester  

https://development.layer.co/layer/sandbox-config/models/gpt-2   