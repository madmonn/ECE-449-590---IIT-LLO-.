#include "program.h"
#include "evaluation.h"

program::program()
{
}

void program::append_expression(int expr_id, const char *op_name, const char *op_type, int inputs[],  int num_inputs)
{
    exprs_.push_back(expression(expr_id, op_name, op_type, inputs, num_inputs));
}

int program::add_op_param_double(const char *key, double value)
{
    exprs_.back().add_op_param_double(key, value);
    return 0;
}

int program::add_op_param_ndarray( const char *key, int dim, size_t shape[], double data[])
{
    exprs_.back().add_op_param_ndarray(key, dim, shape, data);
    return 0;
}

evaluation *program::build()
{
    evaluation *eval = new evaluation(exprs_);
    return eval;
}