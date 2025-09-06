#include <iostream>
#include "tensor.hpp"
#include "layers/linear.hpp"
#include "layers/embedding.hpp"
#include "layers/layerNorm.hpp"
#include "layers/feedForward.hpp"


int main() {
    // -----------------------------
    // Test Linear layer
    // -----------------------------
    std::cout << "🚀 Testing Linear layer..." << std::endl;

    Tensor x = Tensor::Random(4, 3);  // 4 features, batch size 3
    std::cout << "Input x:\n" << x << "\n\n";

    Linear linear_layer(4, 2);        // 4 -> 2
    Tensor y = linear_layer.forward(x);
    std::cout << "Output y:\n" << y << "\n\n";

    // -----------------------------
    // Test Embedding layer
    // -----------------------------
    std::cout << "🚀 Testing Embedding layhher..." << std::endl;

    int vocab_size = 50;
    int embed_dim = 8;
    int max_seq = 10;
    bool sinusoidal = true;

    Embedding emb(vocab_size, embed_dim, max_seq, sinusoidal);

    TensorInt input_indices(5);       // sequence length 5
    input_indices << 3, 12, 7, 25, 0;

    std::cout << "Input indices:\n" << input_indices.transpose() << "\n\n";

    Tensor output = emb.forward(input_indices);
    std::cout << "Embedding output:\n" << output << "\n";

     std::cout << "🚀 Testing LayerNorm..." << std::endl;

    LayerNorm ln(embed_dim);  // normalize along embedding dimension
    Tensor ln_output = ln.forward(output);

    std::cout << "LayerNorm input (Embedding output):\n" << output << "\n\n";
    std::cout << "LayerNorm output:\n" << ln_output << "\n\n";

    std::cout << "✅ All tests executed successfully.\n";
       std::cout << "🚀 Testing FeedForward layer..." << std::endl;

   FeedForward net({4, 5, 3});

    // Example batch: 2 samples, 4 features each
    Tensor x2(1, 4);
    x << 1, 2, 3, 4,
         5, 6, 7, 8;

    Tensor y2 = net.forward(x2);

    std::cout << "Output:\n" << y2 << std::endl;
    return 0;
}
