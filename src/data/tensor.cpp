#include "data/tensor.hpp"
#include <iostream>
#include <set>
#include <algorithm>


Tensor::Tensor(const Eigen::MatrixXd& data, bool requires_grad)
        : data(data), requires_grad(requires_grad) {
        if (requires_grad) grad = Eigen::MatrixXd::Zero(data.rows(), data.cols());
    }

Tensor::Tensor(int rows, int cols, bool requires_grad)
        : data(Eigen::MatrixXd::Zero(rows, cols)), requires_grad(requires_grad) {
        if (requires_grad) grad = Eigen::MatrixXd::Zero(rows, cols);
    }


void Tensor::zero_grad() {
    if (requires_grad) {
        grad.setZero();
    }
}



void Tensor::backward() {
    if (!requires_grad) {
        throw std::runtime_error("Cannot call backward on tensor that doesn't require grad");
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

// Operations
// Here is where we're actually storing the gradiernt formulas
std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {

    

    auto result = std::make_shared<Tensor>(a->data + b->data, 
                                          a->requires_grad || b->requires_grad);
    
    if (result->requires_grad) {
        // we need to know where the c comes from do we update it's dependencies.
        result->dependencies = {a, b};
        // imp: how do we know what grad function looks like? this is where we have that:
        result->grad_fn = std::make_shared<std::function<void()>>([a, b, result_weak = std::weak_ptr<Tensor>(result)]() {
            auto result = result_weak.lock();
            if (!result) return;
            if (a->requires_grad) {
                a->backward_impl(result->grad); // send this along the edge of a 
            }
            if (b->requires_grad){
                b->backward_impl(result->grad); // sending this along the edge of b
            }
        });
    }
    
    return result;
}


std::shared_ptr<Tensor> Tensor::add_broadcast(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {

    // Broadcast addition: each row of xW gets b added to it
    auto result = std::make_shared<Tensor>(
        a->data.rowwise() + b->data.row(0), 
        a->requires_grad || b->requires_grad
    );
    
    if (result->requires_grad) {
        result->dependencies = {a, b};
        result->grad_fn = std::make_shared<std::function<void()>>([a, b, result_weak = std::weak_ptr<Tensor>(result)]() {
            auto result = result_weak.lock();
            if (!result) return;
            if (a->requires_grad) {
                a->backward_impl(result->grad);
            }
            if (b->requires_grad) {
                // Sum gradients across batch dimension for bias
                Eigen::MatrixXd b_grad = result->grad.colwise().sum();
                b->backward_impl(b_grad);
            }
        });
    }
return result;
    }


std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    auto result = std::make_shared<Tensor>(a->data * b->data, 
                                          a->requires_grad || b->requires_grad);

    // actually compute the result
    
    if (result->requires_grad) {
        result->dependencies = {a, b};
        // again set the dependencies 
        result->grad_fn = std::make_shared<std::function<void()>>([a, b, result_weak = std::weak_ptr<Tensor>(result)]() {
            auto result = result_weak.lock();
            if (!result) return;
            if (a->requires_grad) {
                // c wrt to a is just b
                Eigen::MatrixXd grad_a = result->grad * b->data.transpose();
                a->backward_impl(grad_a);
            }
            if (b->requires_grad) {
                // c wrt to b is just a
                Eigen::MatrixXd grad_b = a->data.transpose() * result->grad;
                b->backward_impl(grad_b);
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
            [a, result_weak = std::weak_ptr<Tensor>(result), start_col, width]() {
                auto result = result_weak.lock();
                if (!result) return;
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
        result->grad_fn = std::make_shared<std::function<void()>>([tensors, result_weak = std::weak_ptr<Tensor>(result)]() {
        int col_offset = 0;
        auto result = result_weak.lock();
        if (!result) return;
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




std::shared_ptr<Tensor> Tensor::relu(std::shared_ptr<Tensor> x) {
    Eigen::MatrixXd relu_data = x->data.unaryExpr([](double v){ 
        return std::max(0.0, v); 
    });
    
    auto result = std::make_shared<Tensor>(relu_data, x->requires_grad);
    // basically the same thing we capture the relu. take each element v and apply a relu and then just pass it within the shared ptr
    
    if (result->requires_grad) {
        result->dependencies = {x};
        // only one dependency 
        result->grad_fn = std::make_shared<std::function<void()>>([x, result_weak = std::weak_ptr<Tensor>(result)]() {
            auto result = result_weak.lock();
            if (!result) return;
            if (x->requires_grad) {
                // ReLU derivative is just 1 if x > 0, else 0
                Eigen::MatrixXd relu_grad = x->data.unaryExpr([](double v){ 
                    return v > 0.0 ? 1.0 : 0.0; 
                });
                // we need to apply that for each element v.
                Eigen::MatrixXd grad_x = result->grad.cwiseProduct(relu_grad);
                x->backward_impl(grad_x);
            }
        });
    }
    
    return result;
}

std::shared_ptr<Tensor> Tensor::gelu(std::shared_ptr<Tensor> x) {
    Eigen::MatrixXd gelu_data = x->data.unaryExpr([](double v) {
        double c = std::sqrt(2.0 / M_PI);
        double inner = c * (v + 0.044715 * std::pow(v, 3));
        return 0.5 * v * (1.0 + std::tanh(inner));
    });

    auto result = std::make_shared<Tensor>(gelu_data, x->requires_grad);

    if (result->requires_grad) {
        result->dependencies = {x};
        result->grad_fn = std::make_shared<std::function<void()>>([x, result_weak = std::weak_ptr<Tensor>(result)]() {
            auto result = result_weak.lock();
            if (!result) return;
            if (x->requires_grad) {
                // derivative of gelu approximation
                Eigen::MatrixXd grad_gelu = x->data.unaryExpr([](double v) {
                    double c = std::sqrt(2.0 / M_PI);
                    double inner = c * (v + 0.044715 * std::pow(v, 3));
                    double tanh_inner = std::tanh(inner);
                    double sech2 = 1.0 - tanh_inner * tanh_inner;

                    double term1 = 0.5 * (1.0 + tanh_inner);
                    double term2 = 0.5 * v * sech2 * c * (1 + 3 * 0.044715 * v * v);
                    return term1 + term2;
                });
                Eigen::MatrixXd grad_x = result->grad.cwiseProduct(grad_gelu);
                x->backward_impl(grad_x);
            }
        });
    }

    return result;
}


std::shared_ptr<Tensor> Tensor::transpose_mat(std::shared_ptr<Tensor> a) {

    auto result = std::make_shared<Tensor>(a->data.transpose(), a->requires_grad);

    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>(
            [a, result_weak = std::weak_ptr<Tensor>(result)]() {
            auto result = result_weak.lock();
            if (!result) return;
                if (a->requires_grad) {
                    a->backward_impl(result->grad.transpose());
                }
            }
        );
    }

    return result;
}



// Y= X/scale => dY/dX
std::shared_ptr<Tensor> Tensor::scale_mat(std::shared_ptr<Tensor> a, double scaler) {
    auto result = std::make_shared<Tensor>(a->data * scaler, a->requires_grad);

    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>(
            [a, result_weak = std::weak_ptr<Tensor>(result), scaler]() {
                  auto result = result_weak.lock();
            if (!result) return;
                if (a->requires_grad) {
                    a->backward_impl(result->grad * scaler);
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
        result->grad_fn = std::make_shared<std::function<void()>>([a, result_weak = std::weak_ptr<Tensor>(result)]() {
              auto result = result_weak.lock();
            if (!result) return;
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
    // Broadcast multiply: each row of 'a' multiplied elementwise by b
    auto result = std::make_shared<Tensor>(
        a->data.array().rowwise() * b->data.array().row(0), 
        a->requires_grad || b->requires_grad
    );

    if (result->requires_grad) {
        result->dependencies = {a, b};
        result->grad_fn = std::make_shared<std::function<void()>>([a, b, result_weak = std::weak_ptr<Tensor>(result)]() {
              auto result = result_weak.lock();
            if (!result) return;
            if (a->requires_grad) {
                // dL/da = dL/dout * b
                Eigen::MatrixXd grad_a = result->grad.array().rowwise() * b->data.array().row(0);
                a->backward_impl(grad_a);
            }
            if (b->requires_grad) {
                // dL/db = sum_rows(dL/dout * a)
                Eigen::MatrixXd grad_b = (result->grad.array() * a->data.array()).colwise().sum();
                b->backward_impl(grad_b);
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
        result->grad_fn = std::make_shared<std::function<void()>>([a, result_weak = std::weak_ptr<Tensor>(result)]() {
            
            auto result = result_weak.lock();
            if (!result) return;
            if (a->requires_grad) {
                a->backward_impl(result->grad);
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

std::shared_ptr<Tensor> Tensor::binarize_tensor(std::shared_ptr<Tensor> a) {
    double alpha = a->data.mean(); 
    Eigen::MatrixXd centered = a->data.array() - alpha;
    Eigen::MatrixXd signed_data = centered.unaryExpr([](double v) { return v >= 0 ? 1.0 : -1.0; });

    auto result = std::make_shared<Tensor>(signed_data, a->requires_grad);

    if (result->requires_grad) {
        result->dependencies = {a};
        result->grad_fn = std::make_shared<std::function<void()>>([a, result_weak = std::weak_ptr<Tensor>(result)]() {
              auto result = result_weak.lock();
            if (!result) return;
            if (a->requires_grad) {
                a->backward_impl(result->grad);
            }
        });
    }

    

    return result;
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


void Tensor::detach_graph() {
    grad_fn.reset();
    for (auto& dep : dependencies) {
        if (dep) dep->detach_graph();
    }
    dependencies.clear();


}
