#include "layers/embedding.hpp"
#include <random>
#include <cmath>
#include <iostream>

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


/**
* @brief Forward pass for the embedder.
* 
* Converts a sequence of token indices into embeddings and adds positional embeddings.
* 
* @param input_indices  Vector of token indices of shape [seq_len].
* @return Shared pointer to Tensor of embeddings of shape [seq_len, embed_dim].
*/

std::shared_ptr<Tensor> Embedding::forward(const std::vector<int>& input_indices){
    int seq_len=input_indices.size();

    Eigen::MatrixXd output(seq_len, embed_dim);

    for(int i=0;i<seq_len;i++){
        int token_idx=input_indices[i];
        Eigen::VectorXd token_emb = token_embeddings->data.row(token_idx);

        Eigen::VectorXd pos_emb;
        if (use_sinusoidal) {

            pos_emb = positional_embeddings_matrix.row(i);
        } else {

            pos_emb = positional_embeddings_tensor->data.row(i);
        }

        output.row(i)=token_emb+pos_emb;

    }
    auto result = std::make_shared<Tensor>(output,
    token_embeddings->requires_grad ||
    (!use_sinusoidal && positional_embeddings_tensor &&
     positional_embeddings_tensor->requires_grad));


    if (result->requires_grad) {
        result->dependencies = {token_embeddings};
        if(!use_sinusoidal && positional_embeddings_tensor){
            result->dependencies.push_back(positional_embeddings_tensor);
        }

        result->grad_fn = std::make_shared<std::function<void()>>(
        [token_embeddings=this->token_embeddings,
         pos_tensor=this->positional_embeddings_tensor,
         input_indices,result]() 
         {
            const auto& grad = result->grad; // upstream grad
          
            for (int i = 0; i < input_indices.size(); ++i) {
                token_embeddings->grad.row(input_indices[i]) += grad.row(i);
                if (pos_tensor) {
                    pos_tensor->grad.row(i) += grad.row(i);
                }
            }
         });

    }
    
    return result;


}

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


// Trying to keep the structure same so there will always be a constructor and an initializer.
Embedding::Embedding(int vocab_size, int embed_dim, int max_seq, bool sinusoidal)
    : vocab_size(vocab_size), embed_dim(embed_dim), max_seq(max_seq), use_sinusoidal(sinusoidal){
    initialize_parameters();
}

void Embedding::initialize_parameters() {

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
    if (use_sinusoidal) {
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



void Embedding::zero_grad() {
    token_embeddings->zero_grad();
    if(!use_sinusoidal && positional_embeddings_tensor )
        positional_embeddings_tensor->zero_grad();
}

std::vector<std::shared_ptr<Tensor>> Embedding::parameters() const{
    return {token_embeddings, positional_embeddings_tensor};
}



void Embedding::get_parameters() const {
    std::cout << "=== Embedding Parameters ===" << std::endl;
    
    // Token embeddings
    std::cout << "\n--- Token Embeddings ---" << std::endl;
    token_embeddings->get_data();
    
    // Positional embeddings, only if trainable
    if (!use_sinusoidal && positional_embeddings_tensor) {
        std::cout << "\n--- Positional Embeddings ---" << std::endl;
        positional_embeddings_tensor->get_data();
    }
    
    if (use_sinusoidal) {
        std::cout << "\n--- Positional Embeddings (Sinusoidal, fixed) ---" << std::endl;
        std::cout << positional_embeddings_matrix << std::endl;
    }
}

void Embedding::get_gradients() const {
    std::cout << "=== Embedding Gradients ===" << std::endl;
    
    // Token embeddings gradient
    std::cout << "\n--- Token Embeddings Grad ---" << std::endl;
    token_embeddings->get_grad();
    
    // Positional embeddings gradient, only if trainable
    if (!use_sinusoidal && positional_embeddings_tensor) {
        std::cout << "\n--- Positional Embeddings Grad ---" << std::endl;
        positional_embeddings_tensor->get_grad();
    }
}
