import tiktoken

text = "What is your name"

encoder = tiktoken.get_encoding(encoding_name="cl100k_base")
tokens = encoder.encode(text)
print(tokens)