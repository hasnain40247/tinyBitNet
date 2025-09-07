// #include <iostream>
// #include <catch2/catch_test_macros.hpp>
// #include "layers/linear.hpp"
// #include "tensor.hpp"

// TEST_CASE("Linear forward produces non-empty output", "[linear]") {
//     Linear layer(4, 2);                   // 4 inputs -> 2 outputs
//     Tensor x = Tensor::Random(4, 3);      // batch size 3

//     std::cout << "Input x:\n" << x << "\n\n";

//     Tensor y = layer.forward(x);

//     std::cout << "Output y:\n" << y << "\n\n";

//     REQUIRE(y.size() > 0);                 // output has some content
// }

// TEST_CASE("Linear output shape is correct", "[linear]") {
//     Linear layer(4, 2);
//     Tensor x = Tensor::Random(4, 3);
//     Tensor y = layer.forward(x);

//     std::cout << "Forward pass output shape: " << y.rows() << " x " << y.cols() << "\n";

//     REQUIRE(y.rows() == 2);                // 2 output features
//     REQUIRE(y.cols() == 3);                // batch size 3
// }
