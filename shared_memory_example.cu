#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <cassert>
using namespace std;

class Managed {
public:
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void *ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

void showVectorVals(string label, vector<double> &v) {
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
};

class Neuron;
class Layer;
class Net;


class Neuron : public Managed {
public:
    double output_val;
    // __device__ void feedforward (Layer &prev_layer);
};

    // __device__
    // void Neuron:: feedforward(Layer &prev_layer) {
        
    // };


class Layer : public Managed {
public:
    vector<Neuron *> neurons;
    unsigned size;
    void populate_layer (unsigned size);
    void set_output_vals(vector<double> new_vals);
    __device__ void feedforward(unsigned index, Layer &prev_layer);
};
    
    //Initialize the layer with a certain number of neurons
    void Layer:: populate_layer (unsigned size) {
        this->size = size;
        for (int i=0; i<size; i++) {
            this->neurons.push_back(new Neuron);
        }
    };

    //Directly set the output vals of all neurons in this layer
    void Layer:: set_output_vals(vector<double> new_vals) {
        assert(new_vals.size() == this->neurons.size());
        for (int i = 0; i<new_vals.size(); i++){
            (*(this->neurons[i])).output_val = new_vals[i];
        }
    };

    __device__ 
    void Layer:: feedforward(unsigned neuron_index, Layer &prev_layer) {
        double sum = 0.0;
        // Code breaks here. You can't call vector.size() in a device or global function (I tried both).
        // This is quite sad. The internet says we just have to live with it and use arrays instead.
        for (int i=0; i<prev_layer.neurons.size(); i++) {
            sum += (*(prev_layer.neurons[i])).output_val;
        };
    };


__global__
void parallel_feedforward(Layer &current_layer, Layer &prev_layer) { 
    current_layer.feedforward(threadIdx.x, prev_layer);
};


class Net : public Managed {
public:
    //Vector of pointers to Neuron objects
    vector<Layer *> layers;
    void populate_net (vector<unsigned> &topology);
    void feedforward (vector<double> input_vals);
};

    void Net:: populate_net (vector<unsigned> &topology) {
        Layer* new_layer;
        for (int i = 0; i < topology.size(); i++) {
            new_layer = new Layer;
            (*new_layer).populate_layer(topology[i]);
            this->layers.push_back(new_layer);
        }
    };

    void Net::feedforward(vector<double> input_vals) {
        (*(this->layers.front())).set_output_vals(input_vals);
        for (int i=1; i < (this->layers.size()); i++) {
            int num_threads = (*(this->layers[i])).neurons.size();
            parallel_feedforward <<<1, num_threads>>> (*(this->layers[i]), *(this->layers[i-1]));
            cudaDeviceSynchronize();
        }
    };


int main(void) {
    vector<unsigned> topology;
    unsigned topology_array_form[4] = {2, 4, 4, 1};
    for (int i=0; i<4; i++){
        topology.push_back(topology_array_form[i]);
    }

    vector<double> inputs_1;
    unsigned inputs_1_array_form[2] = {3.0, 4.0};
    for (int i=0; i<2; i++){
        inputs_1.push_back(inputs_1_array_form[i]);
    }

    Net* net = new Net;
    (*net).populate_net(topology);
    (*net).feedforward(inputs_1);

    // showVectorVals("Topology: ", topology);
    return 0;
}