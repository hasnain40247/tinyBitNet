#include "models/transformerBlock.hpp"

// Constructor is already handled via member initializer list in the header
TransformerBlock::TransformerBlock(int embed_dim, int num_heads, int ffn_hidden)
    : ln1(embed_dim), ln2(embed_dim), ln3(embed_dim),
      mha(embed_dim, num_heads), ffn(embed_dim, ffn_hidden)
{
}

// Forward pass
std::shared_ptr<Tensor> TransformerBlock::forward(std::shared_ptr<Tensor> x) {
    // ===== 1️⃣ Pre-Attention LayerNorm =====
    auto x_norm1 = ln1.forward(x);

    // ===== 2️⃣ Multi-Head Attention =====
    auto attn_out = mha.forward(x_norm1);

    // ===== 3️⃣ Residual + LayerNorm after MHA =====
    auto res1 = x_norm1->operator+(attn_out);  // proper residual
    auto block_out1 = ln2.forward(res1);

    // ===== 4️⃣ Feed-Forward =====
    auto ff_out = ffn.forward(block_out1);

    // ===== 5️⃣ Residual + LayerNorm after FFN =====
    auto res2 = block_out1->operator+(ff_out); // proper residual
    auto out = ln3.forward(res2);

    return out;
}

// Collect all trainable parameters
std::vector<std::shared_ptr<Tensor>> TransformerBlock::parameters() const {
    auto params = ln1.parameters();
    auto ln2_params = ln2.parameters();
    auto ln3_params = ln3.parameters();
    auto mha_params = mha.parameters();
    auto ffn_params = ffn.parameters();

    params.insert(params.end(), ln2_params.begin(), ln2_params.end());
    params.insert(params.end(), ln3_params.begin(), ln3_params.end());
    params.insert(params.end(), mha_params.begin(), mha_params.end());
    params.insert(params.end(), ffn_params.begin(), ffn_params.end());
    return params;
}

// Zero all gradients
void TransformerBlock::zero_grad() {
    ln1.zero_grad();
    ln2.zero_grad();
    ln3.zero_grad();
    mha.zero_grad();
    ffn.zero_grad();
}
