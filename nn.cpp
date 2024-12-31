#include <iostream>
#include <set>
#include <cmath>
#include <functional>
#include <vector>
#include <cstdint>
#include <limits>
#include <memory>



class Value {
public:
    double data;
    double grad;
    std::function<void()> backwards = [](){};
    std::set<std::shared_ptr<Value>> children;
    std::string op;
    std::string label;

    Value(double data = 0.0, std::set<std::shared_ptr<Value>> children = {}, std::string parameter_op = "", std::string parameter_label = "")
        : data(data), grad(0.0), children(children), op(parameter_op), label(parameter_label) {}

    void overflow_check(double& out, double lhs, double rhs) {
        // Check for overflow
        if (lhs > 0 && rhs > 0 && out < 0) {
            out = std::numeric_limits<double>::infinity();
        } 

        // Check for underflow
        if (lhs < 0 && rhs < 0 && out > 0) {
            out = -std::numeric_limits<double>::infinity();
        }

        // else continue
    }

    Value operator+(Value& rhs) {
        double out = this->data + rhs.data;
        Value out_val(out);
        
        // Store strong references to both operands
        auto lhs_copy = std::make_shared<Value>(*this);
        auto rhs_copy = std::make_shared<Value>(rhs);
        
        out_val.children = {lhs_copy, rhs_copy};
        out_val.op = "+";
        
        // Capture shared_ptrs by value in lambda
        auto backwards_func = [lhs_copy, rhs_copy]() {
            lhs_copy->grad += 1.0;
            rhs_copy->grad += 1.0;
        };
        
        out_val.backwards = backwards_func;
        return out_val;
    }

    // Addition with a scalar
    Value operator+(double number) {
        Value rhs(number);
        return *this + rhs;
    }

    Value operator*(double number) {
        Value rhs = Value(number);
        return *this * rhs;
    }

    Value operator*(Value& rhs) {
        double out = this->data * rhs.data;
        Value out_val(out);
        
        // Store copies of the original values
        auto lhs_copy = std::make_shared<Value>(*this);
        auto rhs_copy = std::make_shared<Value>(rhs);
        
        out_val.children = {lhs_copy, rhs_copy};
        out_val.op = "*";
        
        auto backwards_func = [lhs_copy, rhs_copy]() {
            lhs_copy->grad += rhs_copy->data * rhs_copy->grad;
            rhs_copy->grad += lhs_copy->data * lhs_copy->grad;
        };
        
        out_val.backwards = backwards_func;
        return out_val;
    }

    Value operator-(double data) {
        return *this + (-1 * data);
    }

    Value operator-(Value& rhs) {
        Value neg_one = Value(-1);

        return (neg_one * rhs) + *this;
    }
    Value operator/(Value& rhs) {
        Value temp = (rhs^-1) * (*this);
        return temp;
    }

    Value tanh() {
        double x = this->data;
        double res = (exp(2*x) - 1)/(exp(2*x) + 1);
        Value out_val(res);
        
        // Store strong reference
        auto self_copy = std::make_shared<Value>(*this);
        out_val.children = {self_copy};
        out_val.op = "tanh";
        
        // Capture shared_ptr by value
        auto backwards_func = [self_copy, res]() {
            self_copy->grad += (1 - pow(res,2)) * self_copy->grad;
        };
        
        out_val.backwards = backwards_func;
        return out_val;
    }

    Value exponential() {
        double x = this->data;
        Value out_val = Value(exp(x), std::set<std::shared_ptr<Value>>({std::make_shared<Value>(*this)}), "exp");
        std::shared_ptr<Value> out_ptr = std::make_shared<Value>(out_val);
        
        auto backwards_func = [this, out_ptr]() {
            this->grad += out_ptr->grad * out_ptr->grad;
        };
        out_val.backwards = backwards_func;
        return out_val;
    }

    Value operator^(double rhs) {
        Value out_val = Value(pow(this->data, rhs), std::set<std::shared_ptr<Value>>({std::make_shared<Value>(*this)}), "pow");
        std::shared_ptr<Value> out_ptr = std::make_shared<Value>(out_val);
        
        auto backwards_func = [this, out_ptr, rhs]() {
            this->grad += (rhs * pow(this->data, rhs - 1)) * out_ptr->grad;
        };
        
        out_val.backwards = backwards_func;
        return out_val;
    }

    void full_backwards() {
        std::vector<Value*> topo;
        std::set<Value*> visited;
        
        std::function<void(Value*)> build_topo;
        build_topo = [&topo, &visited, &build_topo](Value* v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (auto& child : v->children) {
                    build_topo(child.get());
                }
                topo.push_back(v);
            }
        };
        
        build_topo(this);
        this->grad = 1.0;
        
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backwards();
        }
    }

    Value relu() {
        double x = this->data;
        double result = x < 0 ? 0 : x;
        Value out_val(result, std::set<std::shared_ptr<Value>>({std::make_shared<Value>(*this)}), "ReLU");
        std::shared_ptr<Value> out_ptr = std::make_shared<Value>(out_val);
        
        auto backwards_func = [this, out_ptr]() {
            this->grad += (out_ptr->data > 0) * out_ptr->grad;
        };
        
        out_val.backwards = backwards_func;
        return out_val;
    }
};

// Operator overloads for double on the left-hand side
Value operator+(double lhs, Value& rhs) {
    return Value(lhs) + rhs;
}

Value operator-(double lhs, Value& rhs) {
    return Value(lhs) - rhs;
}

Value operator*(double lhs, Value& rhs) {
    return Value(lhs) * rhs;
}

Value operator-(Value& val) {
    Value neg_one(-1.0);
    return neg_one * val;
}

std::ostream& operator<<(std::ostream &strm, Value &val) {
    strm << "Value(data=" << val.data << ", grad=" << val.grad 
         << ", op='" << val.op << "', label='" << val.label << "')";

    // print child data as well
    strm << "\nChildren: ";
    for (const auto& child : val.children) {
        strm << "(" << child->data << " " << child->grad << ")";
    }

    return strm;
}


class Neuron {
    public:
    std::vector<Value> weight;
    Value bias;
    
    Neuron(int nin) {
        for (int i = 0; i < nin; i++) {
            weight.push_back(Value((double)rand() / RAND_MAX * 2.0 - 1.0));
        }
        bias = Value((double)rand() / RAND_MAX * 2.0 - 1.0); 
    }

    Value call(std::vector<Value>& x) {

        Value sum = Value(0.0);
        for(size_t i = 0; i < weight.size(); i++) {
            Value pair = weight[i] * x[i];
            sum = sum + pair;
        }

        Value res = (bias + sum).tanh();

        return res;
    }

    std::vector<Value*> parameters() {
        std::vector<Value*> parameters;
        parameters.reserve(weight.size() + 1);  // Reserve space for weights + bias

        for (auto& w : weight) {  // Note: using & to get reference
            parameters.push_back(&w);
        }
        parameters.push_back(&bias);
        return parameters;
    }
 
};


class Layer {
    public:
    std::vector<Neuron> neurons;
    
    Layer(int nin, int nout) {
        for (int i = 0; i < nout; i++) {
            neurons.push_back(Neuron(nin));
        }
    }

    std::vector<Value> call(std::vector<Value> x) {
        std::vector<Value> res;
        // Reserve space for the results
        res.reserve(neurons.size());
        
        // Use push_back instead of transform
        for (auto& neuron : neurons) {
            res.push_back(neuron.call(x));
        }
        return res;
    }

    std::vector<Value*> parameters() {
        std::vector<Value*> vals;
        for (auto& neuron : neurons) {  // Note: using & to get reference
            auto params = neuron.parameters();
            vals.insert(vals.end(), params.begin(), params.end());
        }
        return vals;
    }

};

class MLP {
    public:
        std::vector<Layer> layers;
        std::vector<int> size;
        
        MLP(int nin, std::vector<int> nouts) {
            size = {nin};

            for (auto layer : nouts) {
                size.push_back(layer);
            }
            
            for (int i = 0; i < size.size() - 1; i++) {
                layers.push_back(Layer(size.at(i), size.at(i + 1)));
            }        
        }

        Value call(std::vector<double> x) {
            
            std::vector<Value> values = {};

            for (auto val : x) {
                values.push_back(Value(val));
            }

            return call(values);
        }

        Value call(std::vector<Value> x) {
            for (auto layer : layers) {
                x = layer.call(x);
            }
            return x.at(0);
        }

    std::vector<Value*> parameters() {
        std::vector<Value*> vals;
        for (auto& layer : layers) {  // Note: using & to get reference
            auto params = layer.parameters();
            vals.insert(vals.end(), params.begin(), params.end());
        }
        return vals;
    }

};


int main(int argc, char* argv[]) {
    
    std::vector<std::vector<double>> x = {
        {1,2,3,4},
        {-1,-2,-3,-4},
        {1,3,4,5},
    };

    std::vector<double> y = {1,-1,1};
    // Create MLP with 4 inputs and hidden layers [4,4,1]
    MLP mlp = MLP(4, {4,4,1});  // Heap allocation
    
    // Training parameters
    int epochs = 100;
    double learning_rate = 0.05;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        Value loss = Value(0.0);
        
        // Calculate loss for each training example
        for (size_t i = 0; i < x.size(); i++) {
            Value pred = mlp.call(x[i]);  // Use -> instead of .
            Value diff = pred - y[i];
            Value diff_squared = diff * diff;
            loss = loss + diff_squared;
        }
        
        // Backward pass
        loss.full_backwards();
        
        // Update parameters
        auto params = mlp.parameters();  // Use -> instead of .
        for (Value* p : params) {
            p->data -= learning_rate * p->grad;
            p->grad = 0;
        }
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss.data << std::endl;
        }
    }
    
}