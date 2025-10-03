# TinyBitNet

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)
[![Eigen](https://img.shields.io/badge/Eigen-3.3%2B-green)](https://eigen.tuxfamily.org/)

**TinyBitNet** is a minimal Transformer implementation in C++ built from scratch. It includes a core transformer block, autograd engine, and basic training utilities for educational purposes and experimentation.

---

## Features

* **Transformer Encoder Block**

  * Pre-LayerNorm → Multi-Head Attention → Residual
  * Pre-LayerNorm → Feed-Forward Network → Residual

* **Multi-Head Attention**: Custom implementation using Eigen matrices.

* **Feed-Forward Network**: Standard 2-layer MLP with optional ReLU activation.

* **Embedding Layer**: Converts token indices to embedding vectors.

* **Autograd Engine**

  * Tracks computation graph using `Tensor` objects
  * Supports forward and backward passes for gradient computation
  * Operations: addition, matmul, ReLU, softmax, slicing, concatenation, scaling, broadcasting

* **Optimizer**: Stochastic Gradient Descent (SGD)

* **Loss Functions**: Mean Squared Error (MSE) implemented

---

## Dependencies

* **C++17** or later
* **[Eigen](https://eigen.tuxfamily.org/)** (v3.3+)
* **CMake** (v3.10+)

---

## Build Instructions

```bash
git clone https://github.com/hasnain40247/tinyBitNet
cd tinybitnet
./run_tinybitnet.sh
```

---

## File Structure

```
tinybitnet/
├── data/
│   └── tensor.hpp          # Tensor class 
├── components/
│   ├── embedding.hpp
│   ├── attention.hpp
│   └── layerNorm.hpp
├── models/
│   └── transformer.hpp     # Transformer block
│   └── feedForward.hpp     # Two-Layer Feed forward block
│   └── transformerBlock.hpp # Transformer 
├── optimizer/
│   └── sgd.hpp             # SGD optimizer
├── criterion/
│   └── mse.hpp            # Losses
```

---

## Usage Example

```cpp
    Transformer model(vocab_size, embed_dim, max_seq, num_heads, ffn_hidden, num_layers);
    // Dummy input
    std::vector<int> input_indices = {2, 5, 3, 2, 7};
    auto X = model.forward(input_indices);
}
```

---

## Example Training Loop (MSE)

```cpp
   SGD optim(model.parameters(), 0.01);
    MSELoss criterion;

    for (int epoch = 0; epoch < 2; ++epoch) {
        optim.zero_grad();
        auto preds = model.forward(input_indices);
        auto loss = criterion.forward(preds, targets);
        loss->backward();
        optim.step();
    }
}
```

---

## Current Status

* [x] Core transformer forward pass implemented
* [x] Multi-head attention, feed-forward, layer norm, embedding implemented
* [x] Autograd engine with backward pass
* [x] MSE loss and SGD optimizer
* [ ] Cross-entropy loss not implemented
* [ ] BitLinear / 1-bit weight support not implemented
* [ ] Tokenizer and dataset reader not implemented

---
