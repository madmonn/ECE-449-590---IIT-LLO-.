#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include<map>
#include<assert.h>

class tensor {
public:
    tensor();
    explicit tensor(double v);
    tensor(int dim, size_t shape[], double data[]);
    
    int get_dim() const;
    double item() const;
    double &item();
    double get_size();

    double at(size_t i) const;
    double at(size_t i, size_t j) const;
    double at(size_t i, size_t j, size_t k, size_t l)const;

    size_t *get_shape_array();
    double *get_data_array();


private:
    std::vector<size_t> shape_;
    std::vector<double> data_;
    size_t N;
};

#endif //TENSOR_H