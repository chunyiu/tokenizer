from regextokenizer import SPECIAL_TOKENS, RegexTokenizer

tokenizer = RegexTokenizer()

print("Before Training:")
print("Vocabulary length:", len(tokenizer.vocab))
print("Merges length:", len(tokenizer.merges))

#specify file to train on
with open('taylorswift.txt', 'r', encoding='utf-8') as file:
    text = file.read()

#specify vocab size
vocab_size = 10000

tokenizer.train(text, vocab_size, verbose=True)
tokenizer.register_special_tokens(SPECIAL_TOKENS)
assert tokenizer.decode(tokenizer.encode(text, "all")) == text
tokenizer.save("trained_tokenizer")

print("After Training:")
print("Vocabulary length:", len(tokenizer.vocab))
print("Merges length:", len(tokenizer.merges))