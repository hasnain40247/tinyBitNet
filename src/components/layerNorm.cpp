#include "components/layerNorm.hpp"
#include <cmath>
// Constructor: initialize gamma = 1, beta = 0
LayerNorm::LayerNorm(int embed_dim_, double eps_)
    : embed_dim(embed_dim_), eps(eps_) 
{
    gamma = std::make_shared<Tensor>(
        Eigen::MatrixXd::Ones(1, embed_dim), true
    );

    beta = std::make_shared<Tensor>(
        Eigen::MatrixXd::Zero(1, embed_dim), true
    );
}

std::shared_ptr<Tensor> LayerNorm::forward(std::shared_ptr<Tensor> x) {
        Eigen::VectorXd mean = x->data.rowwise().mean();
        Eigen::MatrixXd mean_mat = mean.replicate(1, x->data.cols());
        Eigen::MatrixXd diff = x->data - mean_mat;

        Eigen::VectorXd var =
            (diff.array().square().rowwise().sum() / x->data.cols()).matrix();
        Eigen::MatrixXd var_mat = var.replicate(1, x->data.cols());

        Eigen::MatrixXd Xhat = diff.array() / (var_mat.array() + eps).sqrt();
        auto Xhat_tensor = std::make_shared<Tensor>(Xhat, x->requires_grad);

        if (x->requires_grad) {
            Xhat_tensor->dependencies = {x};
            double eps_local = eps; // capture eps safely

            Xhat_tensor->grad_fn = std::make_shared<std::function<void()>>(
                [x, Xhat_tensor, var, eps_local]() {
                    Eigen::MatrixXd H = Xhat_tensor->grad; 
                    int B = H.rows();
                    int D = H.cols();
                    Eigen::MatrixXd dX(B, D);

                    for (int i = 0; i < B; ++i) {
    Eigen::RowVectorXd xhat = Xhat_tensor->data.row(i);
    Eigen::RowVectorXd h = H.row(i);

    double std_inv = 1.0 / std::sqrt(var(i) + eps_local);

    Eigen::RowVectorXd mean_h =
        Eigen::RowVectorXd::Constant(D, h.mean());
    Eigen::RowVectorXd mean_hxhat =
        Eigen::RowVectorXd::Constant(D, (h.array() * xhat.array()).mean());

    dX.row(i) = std_inv * (
        (h.array() - mean_h.array() - (xhat.array() * mean_hxhat.array()))
    ).matrix();
}


                    x->backward_impl(dX);
                }
            );
        }

        auto out = Xhat_tensor->mulB(gamma)->addB(beta);
        return out;
}