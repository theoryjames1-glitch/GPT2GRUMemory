

# 🧠 GPT2 + GRU Memory with Decay

This project is a **theory-to-prototype** experiment: augmenting GPT-2 with a **differentiable recurrent memory** that persists across turns, with **exponential decay** to prevent runaway drift.

The idea:

* Standard GPT-2 has no true memory between turns.
* We inject a **GRUCell-based memory module** into the transformer.
* The memory **updates each step**, but **decays old states** → mimics *biological forgetting*.

---

## 🔹 Features

* **Persistent memory**: conversation state carries forward between prompts.
* **Decay factor**: old information fades smoothly instead of drifting infinitely.
* **Pluggable hooks**: memory modules can be inserted at input, in the middle, and at the top of GPT-2.
* **Hugging Face compatible**: supports `.generate()` with all decoding tricks:

  * `temperature`, `top_p`, `top_k`
  * `repetition_penalty`
  * `no_repeat_ngram_size`
* Works out-of-the-box on all GPT-2 checkpoints (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).

---

## 🔹 Theory

Without decay:

* Memory values keep growing → drift → collapse.

With decay:

* Memory adapts, but old traces fade exponentially.
* Prevents runaway growth.
* Simulates **consolidation + forgetting** in biological memory.

---

## 🔹 Architecture

* `GRUMemoryWithDecay`:

  * GRUCell stabilizes memory updates.
  * Decay factor controls forgetting speed (`0.99` = slow decay, `0.9` = fast decay).

* `GPT2WithGRUMemory`:

  * Subclasses Hugging Face GPT-2 LM.
  * Injects GRU memory hooks at:

    * Input embedding stage
    * Inside transformer blocks
    * At the top layer

---

## 🔹 Example Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# load GPT-2 with memory
from gpt2_gru_memory import GPT2WithGRUMemory
model = GPT2WithGRUMemory.from_pretrained_with_memory("gpt2-xl", decay=0.97).to("cuda")
model.reset(batch_size=1)

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.5,
    no_repeat_ngram_size=3
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## 🔹 Observed Behavior

* **Ephemeral mode** → memory resets each chat, conversations independent.
* **Persistent mode with decay** → memory influences outputs across turns, but fades gradually.
* On small GPT-2 (`117M`), outputs drift into nonsense.
* On **GPT-2 XL (1.5B)**, outputs are coherent and memory effects are visible.

---

## 🔹 Future Work

* Train the GRU memory so it encodes useful features (instead of drifting).
* Extend to modern backbones (GPT-Neo, LLaMA, Mistral).
* Visualize memory slot activations over time.
* Explore memory-augmented transformers as continual learners.

---

## 🔹 References

* [GPT-2 (Radford et al., 2019)](https://openai.com/research/gpt-2)
* [Neural Turing Machines (Graves et al., 2014)](https://arxiv.org/abs/1410.5401)
* [Differentiable Neural Computers (Graves et al., 2016)](https://www.nature.com/articles/nature20101)

---

✦ This repo is just a **proof of concept**: showing that **persistent recurrent memory with decay can be bolted onto GPT-2 and influence generation**.

