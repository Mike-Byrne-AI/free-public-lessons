# check.py for the lesson "Tokens: AI reads chunks, not letters".
# Part A is the snippet printed in the lesson, line for line (it needs: pip install tiktoken).
# Part B checks every other number the lesson states.
# Run it and save what it prints: python check.py > check.out.txt
# Its first lines must be exactly README.md's "What you should see" box: media/tools/allowed_text.py checks that.

# ---- Part A: printed in the lesson ----
# Token numbers in two OpenAI tokenizers, from OpenAI's tiktoken library.
import tiktoken
for name in ["o200k_base", "cl100k_base"]:
    enc = tiktoken.get_encoding(name)
    print(name)
    print(len(enc.encode("How many r's are in strawberry?")), "tokens")
    print("' strawberry' =", enc.encode(" strawberry"))
    print("'strawberry' =", [enc.decode([t]) for t in enc.encode("strawberry")])

# ---- Part B: everything else the lesson states as a number or a check ----
import sys
print()
print("---- Part B ----")
print("tiktoken", tiktoken.__version__, "on Python", sys.version.split()[0])
q = "How many r's are in strawberry?"
for name in ["o200k_base", "cl100k_base"]:
    enc = tiktoken.get_encoding(name)
    ids = enc.encode(q)
    pieces = [enc.decode([t]) for t in ids]
    print(name, "list size:", enc.n_vocab)
    print(name, "the question's tokens:", pieces, ids)
    # How it works: each word after the first carries the space before it.
    words = [p for p in pieces if p.strip().isalpha()]
    print(name, "words after the first that start with a space:",
          sum(p.startswith(" ") for p in words[1:]), "of", len(words) - 1)
for model in ["gpt-4o", "gpt-4"]:
    print("tiktoken maps", model, "to", tiktoken.encoding_for_model(model).name)
print("letters in strawberry:", len("strawberry"), "| r's:", "strawberry".count("r"))
# How it works: two models' list sizes, from each one's config.json, and their sizes in parameters (Hugging Face's
# count of the numbers in the model's files). Meta's own copy of Llama needs a login, so this reads a public copy of it.
import json, urllib.request
def get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)
size = {}
for model in ["unsloth/Meta-Llama-3.1-8B", "moonshotai/Kimi-K3"]:
    cfg = get("https://huggingface.co/" + model + "/raw/main/config.json")
    vocab = cfg["vocab_size"] if "vocab_size" in cfg else cfg["text_config"]["vocab_size"]
    size[model] = get("https://huggingface.co/api/models/" + model)["safetensors"]["total"]
    print(model, "list size:", vocab, "| parameters:", size[model])
print("Kimi K3 has", round(size["moonshotai/Kimi-K3"] / size["unsloth/Meta-Llama-3.1-8B"]), "times the parameters of Llama 3.1 8B")
