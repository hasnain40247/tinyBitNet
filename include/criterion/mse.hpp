#pragma once
#include "data/tensor.hpp"
#include <memory>

class MSELoss {
public:
    MSELoss() {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> preds,
                                    std::shared_ptr<Tensor> targets) {
        int N = preds->data.rows() * preds->data.cols();

        Eigen::MatrixXd diff = preds->data - targets->data;
        double loss_val = diff.array().square().sum() / N;

        auto loss = std::make_shared<Tensor>(
            Eigen::MatrixXd::Constant(1, 1, loss_val), 
            true
        );

        loss->dependencies = {preds, targets};
        loss->grad_fn = std::make_shared<std::function<void()>>([preds, targets, diff, N]() {
            if (preds->requires_grad) {
                Eigen::MatrixXd grad_preds = (2.0 / N) * diff;
                preds->backward_impl(grad_preds);
            }
            if (targets->requires_grad) {
                Eigen::MatrixXd grad_targets = (-2.0 / N) * diff;
                targets->backward_impl(grad_targets);
            }
        });

        return loss;
    }
};
