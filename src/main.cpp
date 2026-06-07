#include <iostream>
#include "Value.h"

int main () {

    // basic neuron

    // inputs
    auto x1 = std::make_shared<Value>(2.0f, "x1");
    auto x2 = std::make_shared<Value>(0.0f, "x2");

    // weights
    auto w1 = std::make_shared<Value>(-3.0f, "w1");
    auto w2 = std::make_shared<Value>(1.0f, "w2");

    // bias
    auto b = std::make_shared<Value>(6.8813735870195432f, "b");

    auto x1w1 = w1 * x1;
    x1w1->label = "x1w1";

    auto x2w2 = w2 * x2;
    x2w2->label = "x2w2";

    auto x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2->label = "x1w1x2w2";

    // compute output
    auto d = x1w1x2w2 + b;
    d->label = "d";

    auto e = exp(2.0f * d);
    e->label = "e";

    // apply tanh
    auto o = (e - 1) / (e + 1);
    o->label = "o";

    std::cout << "o: " << o->data << std::endl;

    o->backward(o);

    exportGraph(o, "graph.dot");

    return 0;
}
