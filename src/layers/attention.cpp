#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "layers/attention.hpp"
#include <random>



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
        // ensure divisibility
        if (embed_dim % num_heads != 0) {
            throw std::invalid_argument("embed_dim must be divisible by num_heads");
        }

        head_dim = embed_dim / num_heads; // this is essentially dk
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0 / std::sqrt(embed_dim));

        auto init = [&](Eigen::MatrixXd& M) {
            M = Eigen::MatrixXd(embed_dim, embed_dim);
            for (int i = 0; i < embed_dim; i++) {
                for (int j = 0; j < embed_dim; j++) {
                    M(i, j) = dist(gen);
                }
            }
        };
        init(W_Q);
        init(W_K);
        init(W_V);
        init(W_O);
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
Eigen::MatrixXd MultiHeadAttention::forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd* mask) {
        int seq_len = X.rows();

        // Linear projections
        Eigen::MatrixXd Q = X * W_Q; // [seq_len, embed_dim]
        Eigen::MatrixXd K = X * W_K; // [seq_len, embed_dim]
        Eigen::MatrixXd V = X * W_V; // [seq_len, embed_dim]

        // Process each head
        std::vector<Eigen::MatrixXd> head_outputs;
        for (int h = 0; h < num_heads; h++) {
            // slice per head
            Eigen::MatrixXd Q_h = Q.middleCols(h * head_dim, head_dim); // [seq_len, head_dim]
            Eigen::MatrixXd K_h = K.middleCols(h * head_dim, head_dim); // [seq_len, head_dim]
            Eigen::MatrixXd V_h = V.middleCols(h * head_dim, head_dim); // [seq_len, head_dim]

            // Scores QK/root(head_dim)
            Eigen::MatrixXd scores = (Q_h * K_h.transpose()) / std::sqrt((double)head_dim); // [seq_len, seq_len]

            // Masking 
            if (mask) {
                // mask: [seq_len, seq_len]
                scores += *mask;
            }

            // Softmax over the masked matrix
            Eigen::MatrixXd attn = softmax(scores); // [seq_len, seq_len]

            // Weighted sum
            Eigen::MatrixXd head = attn * V_h; // [seq_len, head_dim]

            head_outputs.push_back(head);
        }

        // Concat heads
        Eigen::MatrixXd concat(seq_len, embed_dim);
        for (int h = 0; h < num_heads; h++) {
            concat.middleCols(h * head_dim, head_dim) = head_outputs[h];
        }

        // Final linear projection
        return concat * W_O; // [seq_len, embed_dim]
    }

   