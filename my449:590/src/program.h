#ifndef PROGRAM_H
#define PROGRAM_H

#include "expression.h"
using namespace std;

class evaluation;

class program
{
    vector<expression> exprs_;
public:
    program();

    void append_expression(int expr_id, const char *op_name, const char *op_type, int inputs[], int num_inputs);

    // return 0 for success
    int add_op_param_double( const char *key, double value);

    // return 0 for success
    int add_op_param_ndarray( const char *key, int dim, size_t shape[],  double data[]);

    evaluation *build();
}; // class program

#endif // PROGRAM_H