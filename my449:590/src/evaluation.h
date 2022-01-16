#ifndef EVALUATION_H
#define EVALUATION_H

#include "expression.h"
#include "tensor.h"

#include <memory>
#include "eva_op.h"

using namespace std;

class evaluation
{
    map<string, tensor> kwargs_tensor;
public:
    evaluation(const std::vector<expression> &exprs);
    std::vector<expression> exprs_;
    map<int, tensor> variables_;
    
    void add_kwargs_double(const char *key, double value);

    void add_kwargs_ndarray(const char *key, int dim, size_t shape[], double data[]);

    // return 0 for success
    int execute();
    tensor &get_result();

private:
    tensor result_;
    std::vector<std::shared_ptr<eval_op>> ops_;
}; // class evaluation


#endif // EVALUATION_H
