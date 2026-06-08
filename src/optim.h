#ifndef OPTIM_H
#define OPTIM_H

#include "Value.h"
#include <vector>

// Simple Stochastic Gradient Descent optimizer
class SGD {
    public:
        float learning_rate;
        std::vector<std::shared_ptr<Value>> parameters;

        SGD(float learning_rate, const std::vector<std::shared_ptr<Value>>& parameters)
            : learning_rate(learning_rate), parameters(parameters) {}

        void step() {
            for (auto& param : parameters) {
                param->data += -learning_rate * param->grad;
            }
        }
};

#endif // OPTIM_H