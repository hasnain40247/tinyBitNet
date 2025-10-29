#include "data/tensor.hpp"
#include <iostream>
#include <set>
#include <algorithm>



 Tensor::Tensor(const Eigen::MatrixXd& data, bool requires_grad = false, Device device = Device::CPU)
        : data(data), requires_grad(requires_grad), device(device) {
        if (requires_grad) grad = Eigen::MatrixXd::Zero(data.rows(), data.cols());
    }

Tensor::Tensor(int rows, int cols, bool requires_grad = false, Device device = Device::CPU)
        : data(Eigen::MatrixXd::Zero(rows, cols)), requires_grad(requires_grad), device(device) {
        if (requires_grad) grad = Eigen::MatrixXd::Zero(rows, cols);
    }


void Tensor::zero_grad() {
    if (requires_grad) {
        grad.setZero();
    }
}



void Tensor::backward() {
    if (!requires_grad) {
        throw std::runtime_error("Cnnot acall backward on tensor that doesnt require grad");
    }
    
    Eigen::MatrixXd initial_grad = Eigen::MatrixXd::Ones(data.rows(), data.cols());
    backward_impl(initial_grad);
}

void Tensor::backward_impl(const Eigen::MatrixXd& upstream_grad) {
    if (requires_grad) {
        grad += upstream_grad;
    }
    
    if (grad_fn) {
        (*grad_fn)();
    }
}

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {

    if (a->data.size() != b->data.size())
        throw std::runtime_error("Add: tensor size mismatch");

    Eigen::MatrixXd out(a->data.rows(), a->data.cols());

    if (a->device == Device::CPU) {
        out = a->data + b->data;
    } else if (a->device == Device::CUDA) {
        CudaOps::add(a->data.data(), b->data.data(), out.data(), a->data.size());
    } else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(out, a->requires_grad || b->requires_grad, a->device);

    if (result->requires_grad) {
        result->dependencies = {a, b};
        result->grad_fn = std::make_shared<std::function<void()>>([a, b, result]() {
            Eigen::MatrixXd grad_a(a->data.rows(), a->data.cols());
            Eigen::MatrixXd grad_b(b->data.rows(), b->data.cols());

            if (a->requires_grad || b->requires_grad) {
                if (result->device == Device::CPU) {
                    grad_a = result->grad;
                    grad_b = result->grad;
                } else if (result->device == Device::CUDA) {
                    CudaOps::add_backward(grad_a.data(), grad_b.data(), result->grad.data(), result->grad.size());
                } else {
                    throw std::runtime_error("Unsupported device");
                }
            }

            if (a->requires_grad)
                a->backward_impl(grad_a);
            if (b->requires_grad)
                b->backward_impl(grad_b);
        });
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::add_broadcast(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->device != b->device)
        throw std::runtime_error("Device mismatch in Tensor::add_broadcast");

    Eigen::MatrixXd out(a->data.rows(), a->data.cols());

    if (a->device == Device::CPU) {
 
        out = a->data.rowwise() + b->data.row(0);
    } 
    else if (a->device == Device::CUDA) {
        int M = static_cast<int>(a->data.rows());
        int N = static_cast<int>(a->data.cols());
        CudaOps::add_broadcast(
            a->data.data(), b->data.data(), out.data(), M, N
        );
    } 
    else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(
        out, a->requires_grad || b->requires_grad, a->device
    );

    if (result->requires_grad) {
        result->dependencies = {a, b};
        result->grad_fn = std::make_shared<std::function<void()>>([a, b, result]() {
            int M = static_cast<int>(a->data.rows());
            int N = static_cast<int>(a->data.cols());

            if (a->requires_grad) {
                if (a->device == Device::CPU) {
                    a->backward_impl(result->grad);
                } else {
                    Eigen::MatrixXd grad_a(M, N);
                    Eigen::MatrixXd grad_b(1, N);
                    CudaOps::add_broadcast_backward(
                        grad_a.data(), grad_b.data(), result->grad.data(), M, N
                    );
                    a->backward_impl(grad_a);
                    if (b->requires_grad)
                        b->backward_impl(grad_b);
                    return;
                }
            }

            if (b->requires_grad && a->device == Device::CPU) {
                Eigen::MatrixXd b_grad = result->grad.colwise().sum();
                b->backward_impl(b_grad);
            }
        });
    }

    return result;
}


std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    Eigen::MatrixXd out(a->data.rows(), b->data.cols());

    if (a->device == Device::CPU) {
        out = a->data * b->data;
    } else if (a->device == Device::CUDA) {
        CudaOps::matmul(a->data.data(), b->data.data(), out.data(),
                        a->data.rows(), b->data.cols(), a->data.cols());
    } else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(out, a->requires_grad || b->requires_grad, a->device);

    if (result->requires_grad) {
        result->dependencies = {a, b};
        result->grad_fn = std::make_shared<std::function<void()>>([a, b, result]() {
            Eigen::MatrixXd grad_a(a->data.rows(), a->data.cols());
            Eigen::MatrixXd grad_b(b->data.rows(), b->data.cols());

            if (a->device == Device::CPU) {
                if (a->requires_grad)
                    grad_a = result->grad * b->data.transpose();

                if (b->requires_grad)
                    grad_b = a->data.transpose() * result->grad;
            } 
            else if (a->device == Device::CUDA) {

                CudaOps::matmul_backward(
                    a->data.data(),
                    b->data.data(),
                    result->grad.data(),
                    grad_a.data(),
                    grad_b.data(),
                    a->data.rows(),
                    b->data.cols(),
                    a->data.cols()
                );
            } 
            else {
                throw std::runtime_error("Unsupported device");
            }

            if (a->requires_grad)
                a->backward_impl(grad_a);

            if (b->requires_grad)
                b->backward_impl(grad_b);
        });
    }

    return result;
}



std::shared_ptr<Tensor> Tensor::relu(std::shared_ptr<Tensor> x) {
    Eigen::MatrixXd out(x->data.rows(), x->data.cols());

    if (x->device == Device::CPU) {
        out = x->data.unaryExpr([](double v){ return std::max(0.0, v); });
    } else if (x->device == Device::CUDA) {
        CudaOps::relu_forward(x->data.data(), out.data(), x->data.size());
    } else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(out, x->requires_grad, x->device);

    if (result->requires_grad) {
        result->dependencies = {x};
        result->grad_fn = std::make_shared<std::function<void()>>([x, result]() {
            if (x->requires_grad) {

            Eigen::MatrixXd grad_x(x->data.rows(), x->data.cols());

            if (x->device == Device::CPU) {
                Eigen::MatrixXd relu_grad = x->data.unaryExpr([](double v){ return v > 0.0 ? 1.0 : 0.0; });
                grad_x = result->grad.cwiseProduct(relu_grad);
            } else if (x->device == Device::CUDA) {
                CudaOps::relu_backward(x->data.data(), result->grad.data(), grad_x.data(), x->data.size());
            }

            x->backward_impl(grad_x);
            }
        });
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::gelu(std::shared_ptr<Tensor> x) {
    Eigen::MatrixXd out(x->data.rows(), x->data.cols());

    if (x->device == Device::CPU) {
        out = x->data.unaryExpr([](double v) {
            double c = std::sqrt(2.0 / M_PI);
            double inner = c * (v + 0.044715 * v * v * v);  
            return 0.5 * v * (1.0 + std::tanh(inner));
        });
    } else if (x->device == Device::CUDA) {
        CudaOps::gelu(x->data.data(), out.data(), x->data.size());
    } else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(out, x->requires_grad, x->device);

    if (result->requires_grad) {
        result->dependencies = {x};
        result->grad_fn = std::make_shared<std::function<void()>>([x, result]() {
            if (!x->requires_grad) return;

            Eigen::MatrixXd grad_x(x->data.rows(), x->data.cols());

            if (x->device == Device::CPU) {
                Eigen::MatrixXd grad_gelu = x->data.unaryExpr([](double v) {
                    double c = std::sqrt(2.0 / M_PI);
                    double inner = c * (v + 0.044715 * v * v * v);
                    double tanh_inner = std::tanh(inner);
                    double sech2 = 1.0 - tanh_inner * tanh_inner;

                    double term1 = 0.5 * (1.0 + tanh_inner);
                    double term2 = 0.5 * v * sech2 * c * (1 + 3 * 0.044715 * v * v);
                    return term1 + term2;
                });
                grad_x = result->grad.cwiseProduct(grad_gelu);
            } else if (x->device == Device::CUDA) {
                CudaOps::gelu_backward(x->data.data(), result->grad.data(), grad_x.data(), x->data.size());
            }

            x->backward_impl(grad_x);
        });
    }

    return result;
}


std::shared_ptr<Tensor> Tensor::transpose_mat(std::shared_ptr<Tensor> a) {
    Eigen::MatrixXd out(a->data.cols(), a->data.rows());

    if (a->device == Device::CPU) {
        out = a->data.transpose();
    } else if (a->device == Device::CUDA) {
        CudaOps::transpose(a->data.data(), out.data(), a->data.rows(), a->data.cols());
    } else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(out, a->requires_grad, a->device);

    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>([a, result]() {
            if (a->requires_grad) {
                Eigen::MatrixXd grad_a(a->data.rows(), a->data.cols());

                if (a->device == Device::CPU) {
                    grad_a = result->grad.transpose();
                } else {
                    CudaOps::transpose_backward(result->grad.data(), grad_a.data(), a->data.rows(), a->data.cols());
                }

                a->backward_impl(grad_a);
            }
        });
    }

    return result;
}


std::shared_ptr<Tensor> Tensor::scale_mat(std::shared_ptr<Tensor> a, double scaler) {
    Eigen::MatrixXd out(a->data.rows(), a->data.cols());

    if (a->device == Device::CPU) {
        out = a->data * scaler;
    } else if (a->device == Device::CUDA) {
        CudaOps::scale(a->data.data(), out.data(), a->data.rows(), a->data.cols(), scaler);
    } else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(out, a->requires_grad, a->device);

    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>(
            [a, result, scaler]() {
                if (a->requires_grad) {
                    Eigen::MatrixXd grad_a(a->data.rows(), a->data.cols());

                    if (a->device == Device::CPU) {
                        grad_a = result->grad * scaler;
                    } else if (a->device == Device::CUDA) {
                        CudaOps::scale_backward(result->grad.data(), grad_a.data(),
                                                a->data.rows(), a->data.cols(), scaler);
                    } 

                    a->backward_impl(grad_a);
                }
            }
        );
    }

    return result;
}




std::shared_ptr<Tensor> Tensor::softmax_mat(std::shared_ptr<Tensor> a) {

  Eigen::MatrixXd out(a->data.rows(), a->data.cols());
    for (int i = 0; i < a->data.rows(); ++i) {
        double max_val = a->data.row(i).maxCoeff();
        Eigen::VectorXd exps = (a->data.row(i).array() - max_val).exp();
        out.row(i) = exps / exps.sum();
    }

    
    auto result = std::make_shared<Tensor>(out, a->requires_grad);

    // Step 3: define grad_fn if needed
    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>([a, result]() {
            if (a->requires_grad) {
         
                Eigen::MatrixXd grad_input(a->data.rows(), a->data.cols());
                for (int i = 0; i < a->data.rows(); ++i) {
                    Eigen::VectorXd y = result->data.row(i);
                    // Eigen::MatrixXd jac = y.asDiagonal() - y * y.transpose();
                    Eigen::MatrixXd jac = y.asDiagonal().toDenseMatrix() - y * y.transpose();

                    grad_input.row(i) = (jac * result->grad.row(i).transpose()).transpose();
                }
                a->backward_impl(grad_input);
            }
        });
    }

    return result;
}
std::shared_ptr<Tensor> Tensor::mul_broadcast(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    int M = a->data.rows();
    int N = a->data.cols();

    Eigen::MatrixXd out(M, N);

    if (a->device == Device::CPU) {
        // CPU: each row of 'a' multiplied elementwise by 'b'
        out = a->data.array().rowwise() * b->data.array().row(0);
    } else if (a->device == Device::CUDA) {
        // GPU: call CUDA kernel
        CudaOps::mul_broadcast(a->data.data(), b->data.data(), out.data(), M, N);
    } else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(out, a->requires_grad || b->requires_grad, a->device);

    if (result->requires_grad) {
        result->dependencies = {a, b};
        result->grad_fn = std::make_shared<std::function<void()>>([a, b, result, M, N]() {
            Eigen::MatrixXd grad_a(M, N);
            Eigen::MatrixXd grad_b(1, N);

            if (a->device == Device::CPU) {
                if (a->requires_grad) {
                    grad_a = result->grad.array().rowwise() * b->data.array().row(0);
                    a->backward_impl(grad_a);
                }

                if (b->requires_grad) {
                    grad_b = (result->grad.array() * a->data.array()).colwise().sum();
                    b->backward_impl(grad_b);
                }
            } else if (a->device == Device::CUDA) {
                if (a->requires_grad || b->requires_grad) {
                    Eigen::MatrixXd grad_a_host(M, N);
                    Eigen::MatrixXd grad_b_host(1, N);

                    CudaOps::mul_broadcast_backward(
                        a->data.data(), b->data.data(), result->grad.data(),
                        grad_a_host.data(), grad_b_host.data(), M, N
                    );

                    if (a->requires_grad)
                        a->backward_impl(grad_a_host);

                    if (b->requires_grad)
                        b->backward_impl(grad_b_host);
                }
            } else {
                throw std::runtime_error("Unsupported device");
            }
        });
    }

    return result;
}



std::shared_ptr<Tensor> Tensor::quantize_tensor(std::shared_ptr<Tensor> a,double Qb, double eps) {
    double gamma = a->data.cwiseAbs().maxCoeff();
    double eta = a->data.minCoeff();

    Eigen::MatrixXd scaled = ((a->data.array() - eta) * (Qb / (gamma + 1e-8))).matrix();
    Eigen::MatrixXd clipped = scaled.array().min(Qb - eps).max(eps);

    auto result = std::make_shared<Tensor>(clipped, a->requires_grad);

    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>([a, result]() {
            if (a->requires_grad) {
                a->backward_impl(result->grad);
            }
        });
    }

    return result;
}




//auto Q_h = Q->slice_cols(h * head_dim, head_dim);

std::shared_ptr<Tensor> Tensor::slice_cols(std::shared_ptr<Tensor> a,int start_col, int width) {

    Eigen::MatrixXd sliced = a->data.block(0, start_col, a->data.rows(), width);

    auto result = std::make_shared<Tensor>(sliced,a->requires_grad);
    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>(
            [a, result, start_col, width]() {
                if (a->requires_grad) {
           
                    Eigen::MatrixXd grad_parent = Eigen::MatrixXd::Zero(
                        a->data.rows(), a->data.cols());
                    grad_parent.block(0, start_col,
                                      a->data.rows(), width) = result->grad;

                    a->backward_impl(grad_parent);
                }
            }
        );
    }
    return result;
}


// Concatenate a vector of tensors along columns (axis=1)
std::shared_ptr<Tensor> Tensor::concat_cols(const std::vector<std::shared_ptr<Tensor>>& tensors) {
    if (tensors.empty()) {
        throw std::runtime_error("concat_cols: empty tensor list");
    }

    int total_cols = 0;
    int rows = tensors[0]->data.rows();
    bool requires_grad = false;

    // Compute total columns and check if any tensor requires grad
    for (auto& t : tensors) {
        if (t->data.rows() != rows) {
            throw std::runtime_error("concat_cols: all tensors must have the same number of rows");
        }
        total_cols += t->data.cols();
        requires_grad = requires_grad || t->requires_grad;
    }

    // Allocate the concatenated data
    Eigen::MatrixXd concat_data(rows, total_cols);
    int col_offset = 0;
    for (auto& t : tensors) {
        concat_data.block(0, col_offset, rows, t->data.cols()) = t->data;
        col_offset += t->data.cols();
    }

    auto result = std::make_shared<Tensor>(concat_data, requires_grad);

    // Set up grad_fn to slice the gradient back to each tensor
    if (requires_grad) {
        result->dependencies = tensors;
        result->grad_fn = std::make_shared<std::function<void()>>([tensors, result]() {
            int col_offset = 0;
            for (auto& t : tensors) {
                if (t->requires_grad) {
                    Eigen::MatrixXd grad_slice = result->grad.block(0, col_offset, t->data.rows(), t->data.cols());
                    t->backward_impl(grad_slice);
                }
                col_offset += t->data.cols();
            }
        });
    }

    return result;
}


std::shared_ptr<Tensor> Tensor::binarize_tensor(std::shared_ptr<Tensor> a) {
    Eigen::MatrixXd out(a->data.rows(), a->data.cols());

    if (a->device == Device::CPU) {
        double alpha = a->data.mean(); 
        Eigen::MatrixXd centered = a->data.array() - alpha;
        out = centered.unaryExpr([](double v) { return v >= 0 ? 1.0 : -1.0; });
    } 
    else if (a->device == Device::CUDA) {
        CudaOps::binarize_forward(a->data.data(), out.data(),
                                  a->data.rows(), a->data.cols());
    } 
    else {
        throw std::runtime_error("Unsupported device");
    }

    auto result = std::make_shared<Tensor>(out, a->requires_grad, a->device);

    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>([a, result]() {
            if (a->requires_grad) {
                Eigen::MatrixXd grad_a(a->data.rows(), a->data.cols());
                if (a->device == Device::CPU) {
                    grad_a = result->grad;
                } else if (a->device == Device::CUDA) {
                    CudaOps::binarize_backward(result->grad.data(),
                                               grad_a.data(),
                                               a->data.rows(), a->data.cols());
                }
                a->backward_impl(grad_a);
            }
        });
    }

    return result;
}


// since this is a conveience operator it essntially cals a.operator(b) in this case there's a shared from this which has a reference to a and the other is the b
std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
    return add(shared_from_this(), other);
}

std::shared_ptr<Tensor> Tensor::addB(std::shared_ptr<Tensor> other) {
    return add_broadcast(shared_from_this(), other);
}

std::shared_ptr<Tensor> Tensor::mm(std::shared_ptr<Tensor> other) {
    return matmul(shared_from_this(), other);
}



std::shared_ptr<Tensor> Tensor::softmax() {
    return softmax_mat(shared_from_this());
}

std::shared_ptr<Tensor> Tensor::slice(int start_col, int width) {
    return slice_cols(shared_from_this(), start_col, width);
}

std::shared_ptr<Tensor> Tensor::concat(const std::vector<std::shared_ptr<Tensor>>& tensors) {
  
    return concat_cols(tensors);

}

std::shared_ptr<Tensor> Tensor::transpose(){
    return transpose_mat(shared_from_this());
}

std::shared_ptr<Tensor> Tensor::scale(double scaler){
    return scale_mat(shared_from_this(),scaler);
}

std::shared_ptr<Tensor> Tensor::mulB(std::shared_ptr<Tensor> other) {
    return mul_broadcast(shared_from_this(), other);
}

std::shared_ptr<Tensor> Tensor::quantize(double Qb, double eps) {
    return quantize_tensor(shared_from_this(), Qb, eps);
}
std::shared_ptr<Tensor> Tensor::binarize() {
    return binarize_tensor(shared_from_this());
}


void Tensor::shape() const {
    std::cout << "Shape: [" << data.rows() << ", " << data.cols() << "]" << std::endl;
}



void Tensor::get_data() const {
    std::cout << "Data:" << std::endl;
    std::cout << data << std::endl;
}

void Tensor::get_grad() const {
    if (requires_grad) {
        std::cout << "Grad:" << std::endl;
        std::cout << grad << std::endl;
    } else {
        std::cout << "Grad: (requires_grad=false)" << std::endl;
    }
}

void Tensor::get() const {
    std::cout << "=== Tensor ===" << std::endl;
    get_data();
    get_grad();
    shape();
    std::cout << "requires_grad: " << (requires_grad ? "true" : "false") << std::endl;
    std::cout << "==============" << std::endl;
}