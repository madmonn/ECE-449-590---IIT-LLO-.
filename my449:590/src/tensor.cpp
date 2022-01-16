#include "tensor.h"
#include "evaluation.h"

tensor::tensor():data_(1, 0)
{
}

tensor::tensor(double v):data_(1, v)
{
}

tensor::tensor(int dim, size_t shape[], double data[]):
shape_(shape, shape+dim)
{ 
    int N=1;
    for (int i=0; i<dim; ++i)
    {
        N = shape[i]*N;
    }

    data_.assign(data, data+N);
}

int tensor::get_dim() const
{
    return shape_.size();
}

double tensor::item() const{
    assert(shape_.empty());
    return data_[0];
}

double &tensor::item(){
    assert(shape_.empty());
    return data_[0];
}

double tensor::at(size_t i) const {
    assert(get_dim() == 1);
    assert(i < shape_[0]);
    return data_[i];
}

double tensor::at(size_t i, size_t j) const{
    assert(get_dim() == 2);
    assert((i < shape_[0]) && (j < shape_[1]));
    return data_[i*shape_[1]+j];
}

double tensor::at(size_t i, size_t j, size_t k, size_t l)const
{
    assert(get_dim() == 4);
    assert((i < shape_[0]) && (j < shape_[1]) && (k < shape_[2]) && (l < shape_[3]) );
    return data_[ (i*shape_[1]*shape_[2]*shape_[3]) + (j*shape_[2]*shape_[3]) + (k*shape_[3]) + l ];

} 

size_t *tensor::get_shape_array(){
    return shape_.empty()? nullptr: &shape_[0];
}

double *tensor::get_data_array() {
    return &data_[0];
}
double tensor::get_size(){
    return data_.size();
}