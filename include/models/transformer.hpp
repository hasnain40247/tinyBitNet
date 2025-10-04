#pragma once
#include "data/tensor.hpp"
#include "components/embedding.hpp"
#include "models/transformerBlock.hpp"
#include <memory>
#include <vector>

/**
 * Transformer
 * -----------
 * Stacks multiple TransformerBlocks with an embedding layer.
 */
class Transformer {
public:
    /**
     * Constructor
     * @param vocab_size : vocabulary size
     * @param embed_dim  : embedding dimension
     * @param max_seq    : maximum sequence length
     * @param num_heads  : number of attention heads per block
     * @param ffn_hidden : hidden dim for feed-forward layers
     * @param num_layers : number of TransformerBlocks to stack
     */
    Transformer(int vocab_size, int embed_dim, int max_seq,
                int num_heads, int ffn_hidden, int num_layers, bool use_bitlinear_=false);

    std::shared_ptr<Tensor> forward(const std::vector<int>& input_indices);


    std::vector<std::shared_ptr<Tensor>> parameters() const;

    void zero_grad();

private:
    std::shared_ptr<Embedding> embed_layer;
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    std::shared_ptr<BaseLinear> lm_head;  
    bool use_bitlinear;

};
