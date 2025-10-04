#pragma once
#include <memory>
#include <vector>
#include "../components/layerNorm.hpp"
#include "../components/attention.hpp"
#include "../models/feedForward.hpp"
#include "../data/tensor.hpp"


class TransformerBlock {
public:
    /**
     * Constructor
     * @param embed_dim : embedding dimension
     * @param num_heads : number of attention heads
     * @param ffn_hidden : hidden size for feed-forward
     */
    TransformerBlock(int embed_dim, int num_heads, int ffn_hidden,bool use_bitLinear_=false);

    /**
     * Forward pass
     * @param x : input tensor [seq_len, embed_dim]
     * @return output tensor [seq_len, embed_dim]
     */
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);


    std::vector<std::shared_ptr<Tensor>> parameters() const;

 
    void zero_grad();

private:
    LayerNorm ln1, ln2, ln3;
    MultiHeadAttention mha;
    FeedForward ffn;

};
