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


int main(int argc, char* argv[]) {
    
    auto a = Value(-2.0, {}, "", "a");
    auto b = Value(3.0, {}, "", "b");

    auto d = a * b; d.label = 'd';
    auto e = a + b; e.label = 'e';
    auto f = d * e; f.label = 'f';

    f.full_backwards();

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << d << std::endl;
    std::cout << e << std::endl;
    std::cout << f << std::endl;

}