#pragma once
#include "data/tensor.hpp"
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>

class CrossEntropyLoss {
public:
    CrossEntropyLoss() {}

    /**
     * Forward pass: computes the cross-entropy loss
     * @param logits : [seq_len, vocab_size] raw predictions
     * @param targets: [seq_len] indices of the correct next token
     */
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> logits,
                                    const std::vector<int>& targets) {
        int seq_len = logits->data.rows();
        int vocab_size = logits->data.cols();

        Eigen::MatrixXd probs(seq_len, vocab_size);
        Eigen::MatrixXd grad_logits = Eigen::MatrixXd::Zero(seq_len, vocab_size);

        double loss_val = 0.0;

        for (int i = 0; i < seq_len; ++i) {
       
            double max_logit = logits->data.row(i).maxCoeff();
            Eigen::VectorXd exps = (logits->data.row(i).array() - max_logit).exp();
            double sum_exps = exps.sum();
            Eigen::VectorXd softmax = exps / sum_exps;

            probs.row(i) = softmax; 

            int target_idx = targets[i]; //target of token1?
            double p = std::max(softmax(target_idx), 1e-12); 
            loss_val -= std::log(p);

            softmax(target_idx) -= 1.0;
            grad_logits.row(i) = softmax;
        }

        loss_val /= seq_len;
        grad_logits /= seq_len; 

        auto loss = std::make_shared<Tensor>(Eigen::MatrixXd::Constant(1, 1, loss_val), true);

        if (logits->requires_grad) {
            loss->dependencies = {logits};
            loss->grad_fn = std::make_shared<std::function<void()>>([logits, grad_logits]() {
                logits->backward_impl(grad_logits);
            });
        }

        return loss;
    }
};
