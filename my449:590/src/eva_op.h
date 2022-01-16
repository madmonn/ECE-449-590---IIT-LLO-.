#ifndef EVAP_OP_H
#define EVAL_OP_H

#include "tensor.h"
#include <map>
#include <vector>
#include "expression.h"
using namespace std;


typedef map<int, tensor> vars_type;
typedef map<string, tensor> kwargs_type;


class eval_op{
    protected:
        int expr_id_;
        string op_name_;
        vector<int> inputs_;
        map<std::string, double> op_params;
        map<std::string, tensor> op_params_tensor;
    public:
        eval_op(const expression &expr);
        virtual ~eval_op();
        virtual tensor eval(vars_type &variables, const kwargs_type &kwargs) = 0;
};


class eval_const: public eval_op{
        tensor value_;
    public:
        eval_const(const expression &expr);
        tensor eval(vars_type &variables , const kwargs_type &kwargs)override;

};

class eval_input: public eval_op{
    public:
        eval_input(const expression &expr);
        tensor eval(vars_type &variables , const kwargs_type &kwargs)override;
};

class eval_add: public eval_op{
    public:
        eval_add(const expression &expr);
        tensor eval(vars_type &variables , const kwargs_type &kwargs)override;
    };

class eval_sub: public eval_op{
    public:
        eval_sub(const expression &expr);
        tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};

class eval_mul: public eval_op{
    public:
        eval_mul(const expression &expr);
        tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};
class eval_relu: public eval_op{
    public:
        eval_relu(const expression &expr);
        tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};
class eval_flatten: public eval_op{
    public:
        eval_flatten(const expression &expr);
        tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};
class eval_Input2d: public eval_op{
    public:
        eval_Input2d(const expression &expr);
        tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};
class eval_Linear: public eval_op{
    tensor weight;
    tensor bias;
    public:
        eval_Linear(const expression &expr);
        tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};
class eval_Maxpool2d: public eval_op{
    public:
        eval_Maxpool2d(const expression &expr);
        tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};
class eval_Convol2d: public eval_op{
    public:
        eval_Convol2d(const expression &expr);
        tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};


#endif 
