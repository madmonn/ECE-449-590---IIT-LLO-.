#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>

#include "tensor.h"
using namespace std;
class evaluation;

class expression
{
    friend class evaluation;
    int expr_id_;
    string op_name_;
    string op_type_;
    vector<int> inputs_;
    vector<int> input_4d;
    map<string, double> op_params;
    map<string, tensor> op_params_tensor;
    
public:

    int get_id() const;
    string get_op_name() const;
    string get_op_type() const;
    map<string, double> get_op_params() const;
    tensor get_op_param(const char *key) const;
    vector<int> get_inputs()const;
    vector<int> get_data_4d()const;
    

    expression(int expr_id, const char *op_name, const char *op_type, int *inputs, int num_inputs);

    void add_op_param_double(const char *key, double value);

    void add_op_param_ndarray(const char *key, int dim, size_t shape[], double data[]);
}; // class expression

#endif // EXPRESSION_H