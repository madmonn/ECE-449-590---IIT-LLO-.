#include "expression.h"

expression::expression(int expr_id, const char *op_name, const char *op_type, int *inputs, int num_inputs):expr_id_(expr_id), op_name_(op_name),op_type_(op_type), inputs_(inputs, inputs+num_inputs),op_params(), op_params_tensor()
{
}


// #Getter function for expr_id_
int expression::get_id() const
{
    return expr_id_;
}

// Getter function for op_name_
std::string expression::get_op_name() const
{
    return op_name_;
}
// Getter function for op_type
std::string expression::get_op_type() const
 {
    return op_type_;
}
// Getter function for op_params
 std::map<std::string, double> expression::get_op_params() const
{
    return op_params;
}
// Getter function for inputs_
std::vector<int> expression::get_inputs()const
{
    return inputs_;
}
tensor expression::get_op_param(const char *key) const
{
    auto it = op_params_tensor.find(key);
    return it->second;
}

void expression::add_op_param_double(const char *key, double value)
{
    op_params[key] = value;
}

void expression::add_op_param_ndarray(const char *key, int dim, size_t shape[],double data[])
{
    
    tensor *tval = new tensor(dim, shape, data);
    op_params_tensor[key] = *tval;
   
}
