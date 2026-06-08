#ifndef NN_H
#define NN_H

#include "Value.h"
#include <vector>

class Neuron {
    public:
        std::vector<std::shared_ptr<Value>> weights;
        std::shared_ptr<Value> bias;
 
        Neuron(int num_weights) {
            for (int i = 0; i < num_weights; i++) {
                // initialize weights randomly between -1 and 1
                weights.push_back(std::make_shared<Value>(rand() / float(RAND_MAX) * 2 - 1, "w" + std::to_string(i)));
            }

            // initialize bias randomly between -1 and 1
            bias = std::make_shared<Value>(rand() / float(RAND_MAX) * 2 - 1, "b");
        }

        std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
            // act = dot product of weights and inputs + bias
            std::shared_ptr<Value> act = std::make_shared<Value>(0.0f, "act");

            for (int i = 0; i < weights.size(); i++) {
                act = act + (weights[i] * inputs[i]);
            }

            act = act + bias;

            // apply activation function
            return tanh(act);
        }

        std::vector<std::shared_ptr<Value>> parameters() {
            std::vector<std::shared_ptr<Value>> params = weights;
            params.push_back(bias);
            return params;
        }
};

class Layer {
    public:
        std::vector<Neuron> neurons;

        Layer(int num_inputs, int num_neurons) {
            // create num_neurons neurons, each with num_inputs weights
            for (int i = 0; i < num_neurons; i++) {
                neurons.emplace_back(num_inputs);
            }
        }

        std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
            std::vector<std::shared_ptr<Value>> outputs;

            // compute output of each neuron in layer
            for (int i = 0; i < neurons.size(); i++) {
                outputs.push_back(neurons[i](inputs));
            }

            return outputs;
        }

        std::vector<std::shared_ptr<Value>> parameters() {
            std::vector<std::shared_ptr<Value>> params;
            for (int i = 0; i < neurons.size(); i++) {
                auto neuron_params = neurons[i].parameters();
                params.insert(params.end(), neuron_params.begin(), neuron_params.end());
            }
            return params;
        }
};

class MLP {
    public:
        std::vector<Layer> layers;

        MLP(int num_inputs, const std::vector<int>& layer_sizes) {
            int current_inputs = num_inputs;

            // create layers based on layer_sizes
            for (int i = 0; i < layer_sizes.size(); i++) {
                int size = layer_sizes[i];
                layers.emplace_back(current_inputs, size);
                current_inputs = size;
            }
        }

        std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
            std::vector<std::shared_ptr<Value>> outputs = inputs;

            // compute outputs of each layer
            for (int i = 0; i < layers.size(); i++) {
                outputs = layers[i](outputs);
            }

            return outputs;
        }

        std::vector<std::shared_ptr<Value>> parameters() {
            std::vector<std::shared_ptr<Value>> params;
            for (int i = 0; i < layers.size(); i++) {
                auto layer_params = layers[i].parameters();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            }
            return params;
        }
};

#endif // NN_H
