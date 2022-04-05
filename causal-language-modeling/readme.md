### Fine tuning Hugging Face for text classification

#### How to use
Make sure you have the latest version of Layer-SDK

``` !pip install layer-sdk -q ``` 

Then, you can fetch the finetuned model and the tokenizer from Layer and start generating text

```
gpt2 = layer.get_model('gpt2-clm').get_train()
tokenizer = layer.get_model('tokenizer').get_train()
input_sequence = "The farmer planted a lot of crops"
# encode context the generation is conditioned on
input_ids = tokenizer.encode(input_sequence, return_tensors='tf')
output = gpt2.generate(input_ids,max_length=100)
print("Output:\n" + 100 * '-')
print(tokenizer.decode(output[0], skip_special_tokens = True))
```
https://development.layer.co/layer/causal-language-modeling/models/tokenizer  
https://development.layer.co/layer/causal-language-modeling/models/gpt2-clm  