

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
#include "tokenizer.cpp"



int main() {
    std::string file_path = "/Users/hasnainsikora/Projects/tinyBit/src/dataset/haikus.txt";
    auto dataset = load_haikus(file_path, 12);

    std::cout << "Loaded " << dataset.inputs.size() << " haikus.\n";
    std::cout << "Vocab size: " << dataset.vocab.size() << "\n";

    int vocab_size = dataset.vocab.size() + 1;  // +1 for padding (id 0)
    int embed_dim = 16;
    int max_seq = 12;
    int num_heads = 2;
    int ffn_hidden = 32;
    int num_layers = 2;

    Transformer model(vocab_size, embed_dim, max_seq, num_heads, ffn_hidden, num_layers, true);
    SGD optim(model.parameters(), 0.001);
    CrossEntropyLoss criterion;

    int num_epochs = 100;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double total_loss = 0.0;
        int num_samples = 0;

        for (const auto& haiku : dataset.inputs) {
            // skip empty or too-short sequences
            if (haiku.size() < 2) continue;

            // Split input/target (shifted by one)
            std::vector<int> input_indices(haiku.begin(), haiku.end() - 1);
            std::vector<int> target_indices(haiku.begin() + 1, haiku.end());

            optim.zero_grad();

            auto preds = model.forward(input_indices);
            auto loss = criterion.forward(preds, target_indices);

            loss->backward();
            loss->detach_graph();
            optim.step();
            

            total_loss += loss->data(0, 0);
            num_samples++;
            loss.reset();
            preds.reset();
        }

        double avg_loss = total_loss / std::max(1, num_samples);
        std::cout << "Epoch " << epoch << " | Avg Loss: " << avg_loss << std::endl;
    }

    return 0;


    

// int vocab_size = 10;
//     int embed_dim = 8;
//     int max_seq = 5;
//     int num_heads = 2;
//     int ffn_hidden = 16;
//     int num_layers = 2;

//     Transformer model(vocab_size, embed_dim, max_seq, num_heads, ffn_hidden, num_layers,true);

//     SGD optim(model.parameters(), 0.01);

//     std::vector<int> input_indices = {2, 5, 3, 2};  
//     std::vector<int> target_indices = {5, 3, 2, 7}; 

//     CrossEntropyLoss criterion;

//     for (int epoch = 0; epoch < 2; ++epoch) {
//         optim.zero_grad();

//         auto preds = model.forward(input_indices);  

//         auto loss = criterion.forward(preds, target_indices);  
//         loss->backward(); 

//         optim.step();

//         std::cout << "Epoch " << epoch << " | Loss: " << loss->data(0,0) << std::endl;
//     }


//     return 0;
}
