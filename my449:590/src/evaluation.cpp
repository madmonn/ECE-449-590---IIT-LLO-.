#include <assert.h>
#include "evaluation.h"


evaluation::evaluation(const std::vector<expression> &exprs)
:result_(0){
        for (auto &expr: exprs)
    {
        if (expr.get_op_type() == "Input"){
            ops_.push_back(std::make_shared<eval_input>(expr));
            
        }
        else if (expr.get_op_type()== "Const"){
            ops_.push_back(std::make_shared<eval_const>(expr));
        }
        else if (expr.get_op_type() == "Add"){
            ops_.push_back(std::make_shared<eval_add>(expr));
        }
        else if (expr.get_op_type() == "Sub"){
            ops_.push_back(std::make_shared<eval_sub>(expr));
        }
        else if (expr.get_op_type() == "Mul"){
            ops_.push_back(std::make_shared<eval_mul>(expr));
        }
        else if (expr.get_op_type() == "ReLu"){
            ops_.push_back(std::make_shared<eval_relu>(expr));
        }
        else if (expr.get_op_type() == "Flatten"){
            ops_.push_back(std::make_shared<eval_flatten>(expr));
        }
        else if (expr.get_op_type() == "Input2d"){
            ops_.push_back(std::make_shared<eval_Input2d>(expr));
        }
        else if (expr.get_op_type() == "Linear"){
            ops_.push_back(std::make_shared<eval_Linear>(expr));
        }
        else if (expr.get_op_type() == "Maxpool2d"){
            ops_.push_back(std::make_shared<eval_Maxpool2d>(expr));
        }
        else if (expr.get_op_type() == "Convol2d"){
            ops_.push_back(std::make_shared<eval_Convol2d>(expr));
        }
    }
}

void evaluation::add_kwargs_double(const char *key, double value)
{
    kwargs_tensor[key] = tensor(value);
}

void evaluation::add_kwargs_ndarray(const char *key, int dim, size_t shape[], double data[])
{
   
    tensor *tval = new tensor(dim, shape, data);
    kwargs_tensor[key] = *tval;
  
}



int evaluation::execute()
{ 
    // variables_.clear();
    tensor *resT = new tensor();
    for (auto &op:ops_) 
    {
        *resT = op->eval(variables_, kwargs_tensor);
        result_ = *resT;
    }
    return 0;
}
   /* for (auto &expr: exprs_) 
    {
        if (expr.get_op_type() == "Input"){
            variables_[expr.get_id()] = kwargs_[expr.get_op_name()];
            result_=kwargs_[expr.get_op_name()];
        }
    else if (expr.get_op_type() == "Add"){
        double v = variables_[expr.inputs_[0]]+variables_[expr.inputs_[1]];
        result_ = v;
     }
    else if (expr.get_op_type() == "Sub"){
        double v = variables_[expr.inputs_[0]]-variables_[expr.inputs_[1]];
        result_ = v; 
     }
    else if (expr.get_op_type() == "Mul"){
        double v = variables_[expr.inputs_[0]]*variables_[expr.inputs_[1]];
        result_ = v;
    }
    else if (expr.get_op_type() == "Const") {
        variables_[expr.get_id()] = expr.op_params["value"];
        result_ = expr.op_params["value"];
    }
    variables_[expr.get_id()] = result_;
    
    }
    */
  

tensor &evaluation::get_result()
{   
    return result_;
}
