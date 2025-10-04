

#include "components/embedding.hpp"
#include "models/transformerBlock.hpp"
#include "components/layerNorm.hpp"
#include <iostream>
#include "data/tensor.hpp"
#include "components/attention.hpp"
#include "components/linear.hpp"
#include "models/feedForward.hpp"
#include "models/transformer.hpp"
#include "optimizer/optimizer.hpp"
#include "criterion/mse.hpp"
#include "criterion/crossEntropy.hpp"

#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include "components/bitLinear.hpp"
#include "data/tensor.hpp"


int main() {
int vocab_size = 10;
    int embed_dim = 8;
    int max_seq = 5;
    int num_heads = 2;
    int ffn_hidden = 16;
    int num_layers = 2;

    Transformer model(vocab_size, embed_dim, max_seq, num_heads, ffn_hidden, num_layers,true);

    SGD optim(model.parameters(), 0.01);

    std::vector<int> input_indices = {2, 5, 3, 2};  
    std::vector<int> target_indices = {5, 3, 2, 7}; 

    CrossEntropyLoss criterion;

    for (int epoch = 0; epoch < 2; ++epoch) {
        optim.zero_grad();

        auto preds = model.forward(input_indices);  

        auto loss = criterion.forward(preds, target_indices);  
        loss->backward(); 

        optim.step();

        std::cout << "Epoch " << epoch << " | Loss: " << loss->data(0,0) << std::endl;
    }


    return 0;
}
