#ifndef VALUE_H
#define VALUE_H

#include <cmath>
#include <vector>
#include <unordered_set>
#include <fstream>

class Value {
    public:
        float data;
        float grad = 0.0f;
        std::vector<std::shared_ptr<Value>> prev;
        std::string op;
        std::string label;
        std::function<void()> backward_step = [](){};

        Value(float data, std::string label = "") : data(data), label(label) {}

        void backward(const std::shared_ptr<Value>& root) {
            // topological sort
            std::vector<std::shared_ptr<Value>> topo;
            std::unordered_set<const void*> visited;

            std::function<void(const std::shared_ptr<Value>&)> buildTopo = [&](const std::shared_ptr<Value>& node) {
                if (!node || visited.count(node.get())) {
                    return;
                }
                
                // mark node as visited
                visited.insert(node.get());

                // visit children (prev nodes)
                for (const auto& child : node->prev) {
                    buildTopo(child);
                }

                // reset node grad and add to end of topo list
                node->grad = 0.0f;
                topo.push_back(node);
            };

            buildTopo(root);

            // set grad of root node to 1 (dL/dL = 1)
            root->grad = 1.0f;

            // backward pass (on reverse topo order)
            for (auto it = topo.rbegin(); it != topo.rend(); it++) {
                (*it)->backward_step();
            }
        }
};

// GLOBAL OPERATOR OVERLOADS
// Addition
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto res = std::make_shared<Value>(a->data + b->data, "(" + a->label + " + " + b->label + ")");
    res->prev = {a, b};
    res->op = "+";

    res->backward_step = [res, a, b]() {
        a->grad += res->grad; // dL/da = dL/dres * dres/da = res->grad * 1
        b->grad += res->grad; // dL/db = dL/dres * dres/db = res->grad * 1
    };

    return res;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, float b) {
    auto b_ptr = std::make_shared<Value>(b, std::to_string(b));
    return a + b_ptr; // use shared_ptr<Value> + shared_ptr<Value> overload above
}

std::shared_ptr<Value> operator+(float a, const std::shared_ptr<Value>& b) {
    return b + a; // use shared_ptr<Value> + float overload above
}

// Multiplication
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto res = std::make_shared<Value>(a->data * b->data, "(" + a->label + " * " + b->label + ")");
    res->prev = {a, b};
    res->op = "*";

    res->backward_step = [res, a, b]() {
        a->grad += res->grad * b->data; // dL/da = dL/dres * dres/da = res->grad * b->data
        b->grad += res->grad * a->data; // dL/db = dL/dres * dres/db = res->grad * a->data
    };

    return res;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, float b) {
    auto b_ptr = std::make_shared<Value>(b, std::to_string(b));
    return a * b_ptr; // use shared_ptr<Value> * shared_ptr<Value> overload above
}

std::shared_ptr<Value> operator*(float a, const std::shared_ptr<Value>& b) {
    return b * a; // use shared_ptr<Value> * float overload above
}

// Power function
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, float exponent) {
    float p = std::pow(a->data, exponent);

    auto res = std::make_shared<Value>(p, "(" + a->label + " ^ " + std::to_string(exponent) + ")");
    res->prev = {a};
    res->op = "^";

    res->backward_step = [res, a, exponent]() {
        a->grad += res->grad * exponent * std::pow(a->data, exponent - 1); // dL/da = dL/dres * dres/da = res->grad * exponent * a^(exponent-1)
    };

    return res;
}

// Division
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a * pow(b, -1.0f); // use shared_ptr<Value> * shared_ptr<Value> overload and power function overload above
}

// Negation
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a) {
    return a * -1.0f; // use shared_ptr<Value> * float overload above
}

// Subtraction
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a + (-b); // use shared_ptr<Value> + shared_ptr<Value> and negation overloads above
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, float b) {
    return a + (-b); // use shared_ptr<Value> + shared_ptr<Value> and negation overloads above
}

std::shared_ptr<Value> operator-(float a, const std::shared_ptr<Value>& b) {
    return a + (-b); // use shared_ptr<Value> + shared_ptr<Value> and negation overloads above
}

// TanH activation function
std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& a) {
    float t = std::tanh(a->data);

    auto res = std::make_shared<Value>(t, "tanh(" + a->label + ")");
    res->prev = {a};
    res->op = "tanh";

    res->backward_step = [res, a, t]() {
        // float t = std::tanh(a->data);
        a->grad += res->grad * (1 - t * t); // dL/da = dL/dres * dres/da = res->grad * (1 - tanh^2(a->data))
    };

    return res;
}

// Exp function (e^x)
std::shared_ptr<Value> exp(const std::shared_ptr<Value>& a) {
    float e = std::exp(a->data);

    auto res = std::make_shared<Value>(e, "exp(" + a->label + ")");
    res->prev = {a};
    res->op = "exp";

    res->backward_step = [res, a, e]() {
        a->grad += res->grad * e; // dL/da = dL/dres * dres/da = res->grad * exp(a->data)
    };

    return res;
}

// GRAPH VISUALIZATION
void buildDot(const std::shared_ptr<Value>& root, std::ofstream& out, std::unordered_set<const void*>& visited) {
    if (!root || visited.count(root.get())) {
        return;
    }
    
    visited.insert(root.get());

    // node label
    out << (uintptr_t)root.get()
        << " [shape=record, label=\""
        << (root->label.empty() ? "?" : root->label)
        << " | data=" << root->data
        << " | grad=" << root->grad
        << "\"];\n";

    // children
    for (const auto& child : root->prev) {        
        buildDot(child, out, visited);

        out << (uintptr_t)child.get() << " -> " << (uintptr_t)root.get();

        // if current node is an op, label edge or node
        if (!root->op.empty()) {
            out << " [label=\"" << root->op << "\"]";
        }

        out << ";\n";
    }
}

void exportGraph(const std::shared_ptr<Value>& root, const std::string& filename) {
    std::ofstream out(filename);

    out << "digraph G {\n";
    out << "node [shape=box];\n";

    std::unordered_set<const void*> visited;
    buildDot(root, out, visited);

    out << "}\n";
}

#endif // VALUE_H
