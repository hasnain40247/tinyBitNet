#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <functional>
#include <set>   

class Tensor : public std::enable_shared_from_this<Tensor>{
public:
    // We need an eigen double matrix to store the actual data but also one that stores the grad
    Eigen::MatrixXd data;
    Eigen::MatrixXd grad;
    
    // we obv need a flag like pytorch to set the grad as true
    bool requires_grad;
    std::shared_ptr<std::function<void()>> grad_fn;
    std::vector<std::shared_ptr<Tensor>> dependencies;
    
    // Probs will need to set the data and the grads so maybe this constructor can be a way of wrapping data.
    Tensor(const Eigen::MatrixXd& data, bool requires_grad = false);
    Tensor(int rows, int cols, bool requires_grad = false);
    
    // backward will actually do the backprop
    void backward();
    void zero_grad();
    
    // Atleast fromn what I'm reading each operation will need to have some reference
    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> x);
    static std::shared_ptr<Tensor> slice_cols(std::shared_ptr<Tensor> a,int start_col, int width);
    static std::shared_ptr<Tensor> concat_cols(const std::vector<std::shared_ptr<Tensor>>& tensors);

    static std::shared_ptr<Tensor> transpose_mat(std::shared_ptr<Tensor> a);
    static std::shared_ptr<Tensor> scale_mat(std::shared_ptr<Tensor> a,double scaler);
    static std::shared_ptr<Tensor> softmax_mat(std::shared_ptr<Tensor> a);



    
    // Just overloading these so that I can use the above functions in a more readable way
    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> mm(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> slice(int start_col, int width);
    std::shared_ptr<Tensor> concat(const std::vector<std::shared_ptr<Tensor>>& tensors);

    std::shared_ptr<Tensor> transpose();
    std::shared_ptr<Tensor> scale(double scaler);
    std::shared_ptr<Tensor> softmax();




    
    // imma need some getter.
    void shape() const;
    void get_data() const;
    void get_grad() const;
    void get() const;
    

    void backward_impl(const Eigen::MatrixXd& upstream_grad);
    static void topological_sort(std::shared_ptr<Tensor> tensor, 
                                std::vector<std::shared_ptr<Tensor>>& topo_order,
                                std::set<std::shared_ptr<Tensor>>& visited);
};