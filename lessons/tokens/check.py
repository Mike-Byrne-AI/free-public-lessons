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
    # The picture's caption: each word after the first carries the space before it.
    words = [p for p in pieces if p.strip().isalpha()]
    print(name, "words after the first that start with a space:",
          sum(p.startswith(" ") for p in words[1:]), "of", len(words) - 1)
for model in ["gpt-4o", "gpt-4"]:
    print("tiktoken maps", model, "to", tiktoken.encoding_for_model(model).name)
print("letters in strawberry:", len("strawberry"), "| r's:", "strawberry".count("r"))
