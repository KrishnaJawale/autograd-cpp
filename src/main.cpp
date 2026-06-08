#include <iostream>
#include <vector>
#include <cstdlib>
#include "Value.h"
#include "nn.h"
#include "optim.h"

int main() {
    // Basic dataset: (x1, x2) -> y
    std::vector<std::pair<std::vector<float>, float>> dataset = {
        {{0.0f, 0.0f}, 0.0f},
        {{0.0f, 1.0f}, 1.0f},
        {{1.0f, 0.0f}, 1.0f},
        {{1.0f, 1.0f}, 0.0f},
    };

    std::srand(42);

    // 2 inputs -> 8 hidden -> 1 output
    MLP model(2, {8, 1});
    SGD optimizer(0.1f, model.parameters());

    std::cout << "Training MLP on XOR dataset...\n" << std::endl;

    for (int epoch = 0; epoch < 100; epoch++) {
        float total_loss = 0.0f;

        for (const auto& sample : dataset) {
            auto x1 = std::make_shared<Value>(sample.first[0], "x1");
            auto x2 = std::make_shared<Value>(sample.first[1], "x2");
            auto target = std::make_shared<Value>(sample.second, "y");

            auto pred = model({x1, x2})[0];
            auto diff = pred - target;
            auto loss = diff * diff;

            total_loss += loss->data;
            loss->backward(loss);
            optimizer.step();
        }

        if (epoch % 10 == 0) {
            // Every 10 epochs, print loss
            std::cout << "epoch " << epoch << "  loss: " << total_loss / dataset.size() << std::endl;
        }
    }

    std::cout << "\nPredictions after training:" << std::endl;
    for (const auto& sample : dataset) {
        auto x1 = std::make_shared<Value>(sample.first[0], "x1");
        auto x2 = std::make_shared<Value>(sample.first[1], "x2");

        auto pred = model({x1, x2})[0];
        std::cout << "  (" << sample.first[0] << ", " << sample.first[1]
                  << ") -> " << pred->data
                  << "  (target: " << sample.second << ")" << std::endl;
    }

    return 0;
}
