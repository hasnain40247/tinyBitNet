

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






int main() {
int vocab_size = 10;
    int embed_dim = 8;
    int max_seq = 5;
    int num_heads = 2;
    int ffn_hidden = 16;
    int num_layers = 1;

    // ===== 1️⃣ Create model =====
    Transformer model(vocab_size, embed_dim, max_seq, num_heads, ffn_hidden, num_layers);

    // ===== 2️⃣ Collect params & optimizer =====
    SGD optim(model.parameters(), 0.01);

    // ===== 3️⃣ Dummy data =====
    std::vector<int> input_indices = {2, 5, 3, 2, 7};
    Eigen::MatrixXd target_data(5, embed_dim);
    target_data.setRandom();   // dummy target
    auto targets = std::make_shared<Tensor>(target_data, false);

    MSELoss criterion;

    // ===== 4️⃣ Training loop =====
    for (int epoch = 0; epoch < 2; ++epoch) {
        // forward
        auto preds = model.forward(input_indices);

        // loss
        auto loss = criterion.forward(preds, targets);

        // // backward
        loss->backward();

        // // update
        optim.step();
        optim.zero_grad();

        // log
        std::cout << "Epoch " << epoch
                  << " | Loss: " << loss->data(0,0) << std::endl;
    }

    return 0;
}
