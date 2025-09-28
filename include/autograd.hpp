// tiny_autograd.hpp
#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <functional>
#include <stack>
#include <unordered_set>

struct Tensor;
struct Function;

// Tensor is reference-counted (shared_ptr) in the graph for simplicity.
struct Tensor {
    Eigen::MatrixXd data;
    Eigen::MatrixXd grad; // allocated only if requires_grad true (lazy allocate recommended)
    bool requires_grad = false;
    std::shared_ptr<Function> grad_fn; // op that created this tensor (nullptr for leaves)

    Tensor() = default;
    Tensor(const Eigen::MatrixXd& d, bool requires_grad_ = false)
        : data(d), requires_grad(requires_grad_) 
    {
        if (requires_grad) grad = Eigen::MatrixXd::Zero(d.rows(), d.cols());
    }

    // Top-level API: compute gradients wrt this tensor
    // seed_grad is optional; if null, uses ones like scalar backward on sum
    void backward(const Eigen::MatrixXd* seed_grad = nullptr);
    void zero_grad() {
        if (requires_grad) grad.setZero();
    }
};

// Abstract Function base: stores inputs and defines backward interface.
struct Function : std::enable_shared_from_this<Function> {
    // store weak_ptrs to avoid cycles
    std::vector<std::weak_ptr<Tensor>> inputs;
    // Save any tensors or scalars needed for backward
    std::vector<Eigen::MatrixXd> saved; 

    virtual ~Function() = default;

    // Called to compute backward: output_grad is gradient of some scalar-output w.r.t. this function's output.
    virtual void backward(const Eigen::MatrixXd& output_grad) = 0;
};

// Utility: build the topo order starting from a tensor
static void build_topo(const std::shared_ptr<Tensor>& root, std::vector<std::shared_ptr<Function>>& topo) {
    std::stack<std::shared_ptr<Function>> st;
    std::unordered_set<Function*> seen;

    if (!root->grad_fn) return;
    st.push(root->grad_fn);
    while (!st.empty()) {
        auto fn = st.top(); st.pop();
        if (!fn || seen.count(fn.get())) continue;
        seen.insert(fn.get());
        topo.push_back(fn);
        // push upstream functions
        for (auto &in_wp : fn->inputs) {
            if (auto in = in_wp.lock()) {
                if (in->grad_fn && !seen.count(in->grad_fn.get())) st.push(in->grad_fn);
            }
        }
    }
}

// Tensor::backward implementation
inline void Tensor::backward(const Eigen::MatrixXd* seed_grad) {
    // Only support starting from single output tensor for now
    if (!requires_grad && !grad_fn) return;

    // seed gradient: if not provided, default to ones of same shape
    Eigen::MatrixXd initial_grad = seed_grad ? *seed_grad : Eigen::MatrixXd::Ones(data.rows(), data.cols());
    // accumulate gradient on the root tensor
    if (requires_grad) {
        if (grad.size() != initial_grad.size()) grad = Eigen::MatrixXd::Zero(initial_grad.rows(), initial_grad.cols());
        grad += initial_grad;
    }

    // build topo (simple DFS -> topological order roughly)
    std::vector<std::shared_ptr<Function>> topo;
    build_topo(std::make_shared<Tensor>(*this), topo);
    // reverse topo to run backward: from output's creator back to leaves.
    for (auto it = topo.begin(); it != topo.end(); ++it) {} // no-op to clarify
    std::reverse(topo.begin(), topo.end());

    // Map function -> grad of its output. For simplicity, store grads in the output Tensor(s).
    // We'll call each fn->backward with the gradient of its output. The backward will accumulate into input->grad.

    // A more robust impl would map each Tensor to its output grad; here we use saved tensors to find inputs.
    for (auto &fn : topo) {
        // find the output grad for this function:
        // In this small design, assume each Function's output is a single tensor that points to this fn.
        // We'll find that output tensor via scanning inputs' owners - but for simplicity we require functions
        // to read grads from their output node(s) saved previously. To keep short we assume the Function saved
        // output_grad in saved[0] as a placeholder; but better is to store a pointer to the output tensor.
        // For a practical implementation: each Function should hold a weak_ptr to its output Tensor(s).
    }

    // A slightly more explicit and clearer approach: do graph walk using tensors
    // We'll build list of tensors in reverse topological order (this is more direct).
    // For brevity, below is a simpler alternative design that concurrency breaks if used heavily.
    // -- In production, store outputs in Function and maintain map<Function,shared_ptr<Tensor>>.
}
