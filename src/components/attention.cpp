#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>
#include "components/attention.hpp"
#include <random>

#include "data/tensor.hpp"

/**
* Constructor.
* Initializes weight matrices with random values. I am sticking with Xavier's random initialization.
* Upon reading through very few resources, it seems like a good choice without risking vanishing gradients. 
* @param embed_dim_ : embedding dimension
* @param num_heads_ : number of attention heads
*
* Method:
* I'm not doing anything crazy. I am using the Mersenne Twister random engine and a normal distribution just because 
* I read that it's slightly more performant. I also have a lambda that I use to initialize all the Weight vectors with the said random gen.
*/

MultiHeadAttention::MultiHeadAttention(int embed_dim_, int num_heads_)
        : embed_dim(embed_dim_), num_heads(num_heads_) 
    {
       if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    head_dim = embed_dim / num_heads;

    initialize_parameters();
    }


void MultiHeadAttention::initialize_parameters() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0 / std::sqrt(embed_dim));

    auto init_tensor = [&](int rows, int cols) {
        Eigen::MatrixXd data(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data(i, j) = dist(gen);
            }
        }
        return std::make_shared<Tensor>(data, true); // trainable
    };

    W_Q = init_tensor(embed_dim, embed_dim);
    W_K = init_tensor(embed_dim, embed_dim);
    W_V = init_tensor(embed_dim, embed_dim);
    W_O = init_tensor(embed_dim, embed_dim);
}

/**
* Row-wise softmax over the last dimension.
* @param x : [rows, cols] matrix
* @return row-normalized matrix (softmax across columns per row)
*/

Eigen::MatrixXd MultiHeadAttention::softmax(const Eigen::MatrixXd& x) {
        Eigen::MatrixXd y = x;
        for (int i = 0; i < x.rows(); i++) {
            double max_val = x.row(i).maxCoeff();      
            Eigen::VectorXd exps = (x.row(i).array() - max_val).exp();
            y.row(i) = exps / exps.sum();
        }
        return y;
    }


/**
* Forward pass of multi-head attention. At the time of writing this, I already know backpropagation is going to be a pain to implement.
* @param X : input sequence embeddings [seq_len, embed_dim]
* @param mask : optional attention mask [seq_len, seq_len] (default nullptr but I'll set it to inf like the book says)
* @return output embeddings [seq_len, embed_dim]
*
* Method:
* A neat trick that I read about was not handling a separate K Q V matrix for all heads and just initialize one with the 
* embed size and then slice them into subsequent blocks. For each head, we calculate the score -> mask -> weight with values and store it.
* These outputs are then concatenated and multiplied with the weight matrix.
*/
std::shared_ptr<Tensor> MultiHeadAttention::forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> mask) {
    int seq_len = x->data.rows();
    auto Q = x->mm(W_Q); // [seq_len, embed_dim]
    auto K = x->mm(W_K); // [seq_len, embed_dim]
    auto V = x->mm(W_V); // [seq_len, embed_dim]

    std::vector<std::shared_ptr<Tensor>> head_outputs;


    for (int h = 0; h < num_heads; h++) {
        auto Q_h = Q->slice(h * head_dim, head_dim); // [seq_len, head_dim]
        auto K_h = K->slice(h * head_dim, head_dim); // [seq_len, head_dim]
        auto V_h = V->slice(h * head_dim, head_dim); // [seq_len, head_dim]

        // Attention scores: [seq_len, seq_len]
        auto scores = Q_h->mm(K_h->transpose())->scale(1.0 / std::sqrt((double)head_dim));

        if (mask != nullptr) {
            scores = scores->operator+(mask);  // add mask if provided
        }

        // Softmax along rows
        auto attn = scores->softmax();

        // Weighted sum: [seq_len, head_dim]
        auto head = attn->mm(V_h);
        head_outputs.push_back(head);
    }

    auto concat = Tensor::concat_cols(head_outputs);

 
    auto output = concat->mm(W_O); // [seq_len, embed_dim]
    return output;
}

   