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
            // set grad of root node to 1 (dL/dL = 1)
            this->grad = 1.0f;

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

                // add node to end of topo list
                topo.push_back(node);
            };

            buildTopo(root);

            // backward pass (on reverse topo order)
            for (auto it = topo.rbegin(); it != topo.rend(); it++) {
                (*it)->backward_step();
            }
        }
};

// GLOBAL OPERATOR OVERLOADS
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
