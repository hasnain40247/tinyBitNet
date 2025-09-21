#include "layers/embedding.hpp"
#include <random>
#include <cmath>

/**
TODO: Do I worry about padding or should I wait till I'm implementing batches?
 */

/**
* @brief Constructor for the Embedding layer.
* 
* @param vocab_size  Number of unique tokens in the vocabulary.
* @param embed_dim   Dimensionality of the embeddings.
* @param max_seq     Maximum sequence length (used for positional embeddings).
* @param sinusoidal  If true, use sinusoidal positional embeddings (NOT trainable); 
*                    otherwise, learnable positional embeddings (trainable).
*/
Embedding::Embedding(int vocab_size, int embed_dim, int max_seq, bool sinusoidal)
    : vocab_size(vocab_size), embed_dim(embed_dim), max_seq(max_seq), use_sinusoidal(sinusoidal) {

    float limit = std::sqrt(6.0f / (vocab_size + embed_dim));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);

    // TOKEN EMBEDDINGS - ALWAYS TRAINABLE
    // Using Xavier's random initialization here.
    // Apparently this tries to make the variance scale of the weights as close to the inputs and the outputs. This avoids blowing or vanishing gradients.
    
    Eigen::MatrixXd token_weights(vocab_size, embed_dim);
    for (int i = 0; i < token_weights.size(); ++i) {
        token_weights.data()[i] = dist(gen);
    }
    token_embeddings = std::make_shared<Tensor>(token_weights, true); // requires_grad = true!
    
    // POSITIONAL EMBEDDINGS
    if (sinusoidal) {
        // Sinusoidal embeddings - NOT trainable (fixed mathematical functions)
        positional_embeddings_matrix = positional_embedding(max_seq, embed_dim);
        positional_embeddings_tensor = nullptr; // Not trainable
    } else {
        // Learned positional embeddings - TRAINABLE
        Eigen::MatrixXd pos_weights(max_seq, embed_dim);
        for (int i = 0; i < pos_weights.size(); i++) {
            pos_weights.data()[i] = dist(gen);
        }
        positional_embeddings_tensor = std::make_shared<Tensor>(pos_weights, true); // requires_grad = true!
        // positional_embeddings_matrix not used in this case
    }
}

/**
* @brief Forward pass for the embedder.
* 
* Converts a sequence of token indices into embeddings and adds positional embeddings.
* 
* @param input_indices  Vector of token indices of shape [seq_len].
* @return Shared pointer to Tensor of embeddings of shape [seq_len, embed_dim].
*/
std::shared_ptr<Tensor> Embedding::forward(const std::vector<int>& input_indices) {
    int seq_len = input_indices.size();
    
    // Create output matrix [seq_len, embed_dim]
    Eigen::MatrixXd output(seq_len, embed_dim);
    
    for (int i = 0; i < seq_len; ++i) {
        int token_idx = input_indices[i];
        
        // Get token embedding (trainable)
        Eigen::VectorXd token_emb = token_embeddings->data.row(token_idx);
        
        // Get positional embedding
        Eigen::VectorXd pos_emb;
        if (use_sinusoidal) {
            // Use fixed sinusoidal embeddings
            pos_emb = positional_embeddings_matrix.row(i);
        } else {
            // Use trainable positional embeddings
            pos_emb = positional_embeddings_tensor->data.row(i);
        }
        
        // Combine token + positional embeddings
        output.row(i) = token_emb + pos_emb;
    }
    
    return std::make_shared<Tensor>(output, true); // Output needs gradients for backprop
}

/**
* @brief Get all trainable parameters.
* 
* @return Vector of trainable tensors (token embeddings and optionally positional embeddings).
*/
std::vector<std::shared_ptr<Tensor>> Embedding::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    
    // Token embeddings are always trainable
    params.push_back(token_embeddings);
    
    // Positional embeddings are only trainable if not sinusoidal
    if (!use_sinusoidal && positional_embeddings_tensor) {
        params.push_back(positional_embeddings_tensor);
    }
    
    return params;
}

/**
* @brief Zero gradients for all trainable parameters.
*/
void Embedding::zero_grad() {
    token_embeddings->zero_grad();
    
    if (!use_sinusoidal && positional_embeddings_tensor) {
        positional_embeddings_tensor->zero_grad();
    }
}

/**
* @brief Generates sinusoidal positional embeddings.
* 
* These are fixed mathematical functions, not trainable.
* 
* @param max_seq   Maximum sequence length
* @param embed_dim Embedding dimension
* @return Eigen::MatrixXd of shape [max_seq, embed_dim]
*/
Eigen::MatrixXd Embedding::positional_embedding(int max_seq, int embed_dim) {
    Eigen::MatrixXd pe(max_seq, embed_dim);
    
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            float angle = pos / std::pow(10000.0f, 2.0f * (i / 2) / embed_dim);
            // using the same exact formula from the paper

            if (i % 2 == 0)
                pe(pos, i) = std::sin(angle);
            else
                pe(pos, i) = std::cos(angle);
        }
    }
    
    return pe;
}

/**
* @brief Print embedding information for debugging.
*/
void Embedding::get() const {
    std::cout << "=== Embedding Layer Info ===" << std::endl;
    std::cout << "Vocab size: " << vocab_size << std::endl;
    std::cout << "Embed dim: " << embed_dim << std::endl;
    std::cout << "Max seq length: " << max_seq << std::endl;
    std::cout << "Positional type: " << (use_sinusoidal ? "Sinusoidal (fixed)" : "Learned (trainable)") << std::endl;
    
    std::cout << "\nToken embeddings shape: [" << token_embeddings->data.rows() 
              << ", " << token_embeddings->data.cols() << "]" << std::endl;
    
    if (use_sinusoidal) {
        std::cout << "Positional embeddings shape: [" << positional_embeddings_matrix.rows() 
                  << ", " << positional_embeddings_matrix.cols() << "] (fixed)" << std::endl;
    } else {
        std::cout << "Positional embeddings shape: [" << positional_embeddings_tensor->data.rows() 
                  << ", " << positional_embeddings_tensor->data.cols() << "] (trainable)" << std::endl;
    }
    
    std::cout << "Total trainable parameters: " << parameters().size() << std::endl;
    std::cout << "============================" << std::endl;
}