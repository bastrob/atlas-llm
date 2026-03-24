# 🧠 LLM Architecture Playground

A research-oriented, from-scratch implementation of modern Large Language Model (LLM) architectures.

This repository is designed to explore, reproduce, and extend architectures such as LLaMA, GPT, and Qwen with a strong focus on modularity, correctness, and systems-level understanding.

---

## 🚀 Goals

- Re-implement modern LLM architectures from scratch (no high-level frameworks)
- Build a modular transformer core reusable across architectures
- Experiment with architectural variations and trade-offs
- Develop research intuition through controlled experiments

---

## 🧱 Design Principles

### 1. Modularity First
Each component is designed to be swappable:
- Attention mechanisms
- Positional encodings
- Normalization layers
- Feedforward blocks

This allows rapid experimentation without rewriting the full model.

---

### 2. Faithful Implementations
Architectures are implemented as close as possible to original specifications:
- No hidden abstractions
- Minimal reliance on external libraries
- Explicit tensor operations

---

### 3. Research-Oriented Engineering
The goal is not just to run models, but to understand:
- Why architectural choices work
- How they impact training and inference
- What trade-offs they introduce

---

### 4. Inference-Aware Design
All components are designed with generation in mind:
- KV-cache support
- Efficient attention computation
- Minimal memory overhead

---

## 🏗️ Current Scope

### ✅ Core Transformer (Decoder-Only)
- Pre-norm architecture
- Causal self-attention
- Residual connections

### 🔄 Implementations in Progress
- RMSNorm
- Rotary Positional Embedding (RoPE)
- SwiGLU feedforward network
- KV-cache compatible attention

---

## 🧩 Planned Architectures

### LLaMA-style Models
- RMSNorm
- RoPE
- SwiGLU
- Bias-free linear layers

### GPT-style Models
- LayerNorm variants
- Absolute positional embeddings (optional)
- Inference optimizations

### Qwen-style Variants
- Long-context adaptations
- Tokenization strategies

---

## 🔬 Experimental Tracks

### Attention Variants
- Multi-head attention (baseline)
- Grouped-query attention (GQA)
- Multi-query attention (MQA)
- FlashAttention (future)

### Beyond Transformers
- State Space Models (SSM)
- Mamba-like architectures

### Scaling & Systems
- KV-cache strategies
- Memory optimization
- Throughput benchmarking

---

---

## 🧪 Philosophy

This repository is not meant to be:
- A production framework
- A high-level API wrapper

It is a **learning and research tool**, focused on:
- clarity over abstraction
- control over convenience

---

## ⚠️ Constraints

- No use of high-level transformer libraries (e.g., HuggingFace internals)
- PyTorch only for tensor operations
- All architectural components implemented manually

---

## 📌 Roadmap

- [ ] Minimal LLaMA block (baseline)
- [ ] Full decoder stack
- [ ] Text generation with KV-cache
- [ ] Attention benchmarking suite
- [ ] MoE implementation
- [ ] Transformer vs SSM comparison

---

## 🧠 References

- Attention Is All You Need
- LLaMA
- GPT family
- Qwen


---

## 📜 License

MIT