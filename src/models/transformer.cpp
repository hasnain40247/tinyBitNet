#include "models/transformer.hpp"
#include "components/linear.hpp"
#include "components/bitLinear.hpp"
Transformer::Transformer(int vocab_size, int embed_dim, int max_seq,
                         int num_heads, int ffn_hidden, int num_layers,bool use_bitlinear_): use_bitlinear(use_bitlinear_)
{
    embed_layer = std::make_shared<Embedding>(vocab_size, embed_dim, max_seq, false);

    for (int i = 0; i < num_layers; ++i) {
        blocks.push_back(std::make_shared<TransformerBlock>(embed_dim, num_heads, ffn_hidden, use_bitlinear_));
    }
    if(use_bitlinear)
        lm_head = std::make_shared<BitLinear>(embed_dim, vocab_size);
    else
        lm_head = std::make_shared<Linear>(embed_dim, vocab_size);



}

std::shared_ptr<Tensor> Transformer::forward(const std::vector<int>& input_indices) {
    auto x = embed_layer->forward(input_indices);
    for (auto& block : blocks) {
        x = block->forward(x);
    }

    x=lm_head->forward(x);
    // x->shape();

    return x;
}


std::vector<std::shared_ptr<Tensor>> Transformer::parameters() const {
    auto params = embed_layer->parameters();
    for (const auto& block : blocks) {
        auto block_params = block->parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }
    return params;
}


void Transformer::zero_grad() {
    embed_layer->zero_grad();
    for (auto& block : blocks) {
        block->zero_grad();
    }
}
