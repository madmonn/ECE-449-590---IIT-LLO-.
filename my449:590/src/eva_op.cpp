#include "eva_op.h"
#include <vector>
using namespace std;

eval_op::~eval_op()
{

}
eval_op::eval_op(const expression &exp)
{
    expr_id_ = exp.get_id();
    op_name_ = exp.get_op_name();
    op_params = exp.get_op_params();
    inputs_ = exp.get_inputs();
}

//const
eval_const::eval_const(const expression &expr): eval_op(expr),value_(expr.get_op_param("value"))
{

}
tensor eval_const::eval(vars_type &variables, const kwargs_type &kwargs)
{
    variables[expr_id_] = value_;
    return value_;
}

//input
eval_input::eval_input(const expression &expr):eval_op(expr){}
tensor eval_input::eval(vars_type &variables, const kwargs_type &kwargs)
{
    
    auto it = kwargs.find(op_name_);
    variables[expr_id_] = it->second;
    return it->second;

}


//addition
eval_add::eval_add(const expression &expr):eval_op(expr){}
tensor eval_add::eval(vars_type &variables, const kwargs_type &kwargs )
{
    tensor input_1 = variables.at(inputs_[0]);
    tensor input_2 = variables.at(inputs_[1]);
    std::vector<double> result;
    for(int i =0; i< input_1.get_size(); ++i)
    {
        double sum_array = input_1.get_data_array()[i] + input_2.get_data_array()[i];
        result.push_back(sum_array);
    }
    variables[expr_id_] = tensor(input_1.get_dim(),input_1.get_shape_array(),&result[0]);
    return variables[expr_id_];
}
//subraction
eval_sub::eval_sub(const expression &expr):eval_op(expr){}
tensor eval_sub::eval(vars_type &variables, const kwargs_type &kwargs )

{
    
    tensor input_1 = variables.at(inputs_[0]);
    tensor input_2 = variables.at(inputs_[1]);
    std::vector<double> result;
    for(int i =0; i< input_1.get_size(); ++i)
    {
        double sub_array = input_1.get_data_array()[i] - input_2.get_data_array()[i];
        result.push_back(sub_array);
    }
    variables[expr_id_] = tensor(input_1.get_dim(),input_1.get_shape_array(),&result[0]);
    return variables[expr_id_];

}



//multipliction
eval_mul::eval_mul(const expression &expr):eval_op(expr){}
tensor eval_mul::eval(vars_type &variables, const kwargs_type &kwargs )
{
    tensor input_1 = variables.at(inputs_[0]);
    tensor input_2 = variables.at(inputs_[1]);
    vector<double>result;

    // Value * Tensor
    if(input_1.get_dim()==0)
    {
       double input_x = input_1.item();
       for(int i =0; i< input_2.get_size(); ++i)
       {
           double x = input_x * input_2.get_data_array()[i];
           result.push_back(x);
       }
    variables[expr_id_] = tensor(input_2.get_dim(),input_2.get_shape_array(),&result[0]);
    return variables[expr_id_];
        
    }

    // Tensor* value
    else if(input_2.get_dim()==0){
        double input_x = input_2.item();
       for(int i =0; i< input_1.get_size(); ++i)
       {
           double x = input_x * input_1.get_data_array()[i];
           result.push_back(x);
       }
    variables[expr_id_] = tensor(input_1.get_dim(),input_1.get_shape_array(),&result[0]);
    return variables[expr_id_];
    }

    // Matrix *Matrix
    else
    {

    // #initializing the result of matrix result = 0
    for(size_t i=0; i < input_1.get_shape_array()[0]; ++i)
        for(size_t j=0; j < input_2.get_shape_array()[1]; ++j)
        {
            result.push_back(0);
            
        }
    // multiplying matrix Input1, Input2 and storing inside the result
    for(size_t i=0; i < input_1.get_shape_array()[0]; ++i)
        for(size_t j=0; j < input_2.get_shape_array()[1]; ++j)
            for(size_t k=0; k < input_1.get_shape_array()[1]; ++k)
            {
                result[i * input_2.get_shape_array()[1] + j] += input_1.at(i,k) * input_2.at(k,j);
            }
    size_t * result_shape = input_1.get_shape_array();
    result_shape[1] = input_2.get_shape_array()[1];
    variables[expr_id_] = tensor(2,result_shape, &result[0]);
    return variables[expr_id_];
    }
  
}

//Relu
eval_relu::eval_relu(const expression &expr):eval_op(expr){}
tensor eval_relu::eval(vars_type &variables, const kwargs_type &kwargs ){
    tensor input_1 = variables.at(inputs_[0]);
    std::vector<double> result;
    for(int i =0; i < input_1.get_size();++i){
        if(input_1.get_data_array()[i] < 0)
        {
             result[i] = 0;
         }
         else
         {
             result[i] = input_1.get_data_array()[i];
         }
        
      }
    variables[expr_id_] = tensor(input_1.get_dim(),input_1.get_shape_array(),&result[0]);
     return variables[expr_id_];

 }
//Flatten
eval_flatten::eval_flatten(const expression &expr):eval_op(expr){}
tensor eval_flatten::eval(vars_type &variables, const kwargs_type &kwargs )
 {

     tensor input_1 = variables.at(inputs_[0]);
     size_t N = input_1.get_shape_array()[0];
     size_t C = input_1.get_shape_array()[1];
     size_t H = input_1.get_shape_array()[2];
     size_t W = input_1.get_shape_array()[3];
     size_t result_shape[2] = {N, C*H*W};
     variables[expr_id_] = tensor(2.0, &result_shape[0], input_1.get_data_array());
     return variables[expr_id_];

 }

// Input2d
eval_Input2d::eval_Input2d(const expression &expr):eval_op(expr){}
tensor eval_Input2d::eval(vars_type &variables, const kwargs_type &kwargs )
 {
     tensor input2d =  kwargs.at(op_name_);
     size_t N = input2d.get_shape_array()[0];
     size_t H = input2d.get_shape_array()[1];
     size_t W = input2d.get_shape_array()[2];
     size_t C = input2d.get_shape_array()[3];
     vector<double>result(N*C*H*W,0);
     for(size_t n=0; n < N; ++n)
         for(size_t c=0; c < C; ++c)
             for(size_t h = 0; h < H;++h)
                 for(size_t w=0; w < W; ++w)
                 {
                     result[(n*C*H*W) + (c*H*W) + (h*W) + w]  = input2d.get_data_array()[(n*H*W*C)+(h*W*C)+(w*C)+c];
                 }
 size_t result_shape[4] = {N, C, H, W};
 variables[expr_id_] = tensor(4.0,&result_shape[0],&result[0]);
 return variables[expr_id_];
 }


//Linear
eval_Linear::eval_Linear(const expression &expr): eval_op(expr),weight(expr.get_op_param("weight")),bias(expr.get_op_param("bias"))
 {

 }
tensor eval_Linear::eval(vars_type &variables, const kwargs_type &kwargs )
 {
     tensor linear_input = variables.at(inputs_[0]);
     size_t N = linear_input.get_shape_array()[0];
     size_t O = weight.get_shape_array()[0];
     size_t I = weight.get_shape_array()[1];
     double result =  0;
     vector<double> result2;
     for (size_t n = 0; n<N; ++n){
         for (size_t o = 0; o<O; ++o){
             for (size_t i = 0; i<I; ++i)
             {
                 result += weight.at(o,i) * linear_input.at(n, i);
             }
         result2.push_back(result + bias.at(o));
         result = 0;
         }
     }
     size_t result_shape[2] = {N,O};
     variables[expr_id_] = tensor(2, result_shape, &result2[0]);
     return variables[expr_id_];
 }

//conv2d
eval_Convol2d::eval_Convol2d(const expression &expr):eval_op(expr){}
tensor eval_Convol2d::eval(vars_type &variables, const kwargs_type &kwargs )
 {
     return variables[expr_id_];

 }

//maxpool2d
eval_Maxpool2d::eval_Maxpool2d(const expression &expr):eval_op(expr){}
tensor eval_Maxpool2d::eval(vars_type &variables, const kwargs_type &kwargs )
 {
     return variables[expr_id_];
 }



