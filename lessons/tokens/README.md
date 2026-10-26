[← All lessons](https://mike-byrne-ai.github.io/free-public-lessons/) · Released 2026-10-26 · Checked 2026-10-25

# Tokens: AI reads chunks, not letters

AI doesn't read your text letter by letter. It sees numbered chunks of text called tokens.

## The short answer

A tokenizer cuts your text into chunks called tokens. A token can be any piece of text from a letter to a full word, even characters or chunks of text, like 'ing'. The model gets each token as a number from a fixed list. In OpenAI's o200k_base tokenizer, "How many r's are in strawberry?" is 8 tokens.

## The picture

![A question about strawberry, cut into its tokens by OpenAI's o200k_base tokenizer. Each word after the first is marked with the space in front of it. The token for strawberry, with its space, has one number. Without the space, the word splits into three pieces.](picture.svg)

## How it works

A tokenizer is a program that builds the fixed list from text before the model is trained, then turns your text into numbered tokens every time you use the model. Common words get a token of their own, and rare words are built from smaller pieces (Sennrich et al., section 3.2). OpenAI's o200k_base list has about 200,000 tokens, and tiktoken uses it for GPT-4o. The older cl100k_base list has about 100,000, and tiktoken uses it for GPT-4.

Each model's makers choose the size of its list. Meta's Llama 3.1 8B, a small model, has about 128,000 tokens on its list. Moonshot's Kimi K3, one of the biggest models anyone can download, is about 350 times bigger, but its list has only about 164,000.

In the question, each word after the first carries the space before it. That is why " strawberry" starts with a space, and with it the word is one token. Without the space, the word splits into three tokens. In o200k_base they are "st", "raw" and "berry". In cl100k_base they are "str", "aw" and "berry", and " strawberry" is token 73700. Each tokenizer cuts text its own way.

So the model is never handed the ten letters of strawberry to count. It is handed one number, or in the case of strawberry typed without a space, it receives three numbers, one for each token.

## Try it

Paste this into Python 3, after `pip install tiktoken`. The first run downloads each tokenizer's list, so it needs the internet.

```python
# Token numbers in two OpenAI tokenizers, from OpenAI's tiktoken library.
import tiktoken
for name in ["o200k_base", "cl100k_base"]:
    enc = tiktoken.get_encoding(name)
    print(name)
    print(len(enc.encode("How many r's are in strawberry?")), "tokens")
    print("' strawberry' =", enc.encode(" strawberry"))
    print("'strawberry' =", [enc.decode([t]) for t in enc.encode("strawberry")])
```

What you should see:

```
o200k_base
8 tokens
' strawberry' = [101830]
'strawberry' = ['st', 'raw', 'berry']
cl100k_base
8 tokens
' strawberry' = [73700]
'strawberry' = ['str', 'aw', 'berry']
```

Both tokenizers make 8 tokens of the question, but they number and split the word differently.

## Sources and checks

- OpenAI, tiktoken (github.com/openai/tiktoken). The two tokenizers, and which model uses each: tiktoken/model.py.
- Sennrich et al. (2016), Neural Machine Translation of Rare Words with Subword Units. How a token list is built from text: section 3.2.
- Meta, Llama 3.1 8B: its config.json (vocab_size), read from a public copy (huggingface.co/unsloth/Meta-Llama-3.1-8B), since Meta's own copy needs a login.
- Moonshot AI, Kimi K3 (huggingface.co/moonshotai/Kimi-K3): its config.json (vocab_size) and model card (2.8T parameters).

Computed by check.py: the 8 tokens, the token numbers 101830 and 73700, the three-way splits, the two list sizes, the ten letters, the Llama and Kimi list sizes and parameter counts (read from Hugging Face), and which model tiktoken maps to each list.

**How this was checked** (2026-10-07, the Llama and Kimi numbers 2026-10-25): check.py ran tiktoken 0.14.0 on Python 3.11. Which model uses which list is tiktoken's own table, not OpenAI's model pages.

## License

Words, picture, Reel and music: CC0 1.0. Code: MIT-0. Copy, change and share anywhere, with no credit needed.
