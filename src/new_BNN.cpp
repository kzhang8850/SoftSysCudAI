#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include "training_reader.h"



#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 32
#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define NUM_HIDDEN_LAYERS 1

using namespace std;


//----------------------------------Neural Declarations-------------------------
class Connection;
class Neuron;
class Layer;
class Network;

//--------------------------------Global Declarations---------------------------

void showVectorVals(string label, double *v, int length);
void neuron_global_feed_forward(Neuron *n, double *sum, Layer &prev_layer);
void neuron_global_sum_DOW(Neuron *n, double *sum, Layer &next_layer);
void neuron_global_update_input_weights(Neuron *n, Layer &prev_layer);
void net_global_feed_forward(Layer &layer, Layer &prev_layer);
void net_global_update_weights(Layer &layer, Layer &prev_layer);
void net_global_backprop(Layer &hidden_layer, Layer &next_layer);


//-------------------------------Net Class Initializations----------------------

class Connection
{
    // This class represents the connection
    //  between two neurons on different layers.
    //  Each neuron has one connection for every
    //  neuron on the previous layer as well as
    //  every neuron on the next layer.
public:
    double weight; // Used to determine how much emphasis this connection should have
    double delta_weight; // Used in back propagation to alter the weight for the next iteration
};


class Neuron
{
    // This class represents the nodes of the BNN.
    //  Each neuron gathers and sums the inputs of
    //  all the neurons previously, altered by the
    //  connection weights, and then pushes forward
    //  the newly calculated result to every neuron
    //  in the next layer.
public:
    Neuron();
    Neuron(int num_neurons, int num_connections);
    void set_output(double val){output = val;}
    double get_output(void) const {return output;}
    void feed_forward(Layer &prev_layer);
    void calculate_output_gradient(double target_val);
    void calculate_hidden_gradients(Layer &next_layer);
    void update_input_weights(Layer &prev_layer);
    double output;
    Connection* output_weights;
    unsigned my_index;
    double gradient;
    static double lr;
    static double momentum;
private:

    static double transfer_function(double x);
    static double transfer_function_derivative(double x);
    static double init_weight(void) {return rand()/double(RAND_MAX);} // initial weights are random, and will migrate over time
    double sum_DOW(Layer &next_layer);


};
double Neuron::lr = 0.15;
double Neuron::momentum = 0.5;


class Layer
{
    // This class represents a set of neurons. Most
    //  of these layers are 'hidden', with only the
    //  input and output layers being exposed.
    //  They operate largely as arrays for neurons.
public:
    Layer();
    Layer(int num_neurons, int num_connections);
    Neuron* layer;
    int length;
};


class Network
{
    // The network represents the entirity of the
    //  algorithm, containing all the layers and
    //  the functions needed to generally control
    //  the forward and backward operations. This
    //  is the interface to the algorithm.
public:
    Network();
    void feed_forward(double *input_vals, int input_vals_length);
    void back_prop(double * target_vals, int target_length);

    void get_results(double *result_vals, int result_length);
    double get_RAE() const { return RAE; }

private:
    Layer *layers;
    double error;
    double RAE;
    static double RAS;

};
double Network::RAS = 100.0; //Number of training samples to average over


//------------------------------Global Functions--------------------------------

void showVectorVals(string label, double *v, int length)
{
    // Prints out the label and the values of the given vector.
    //  This is actually done on host.
    cout << label << " ";
    for (unsigned i = 0; i < length; ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
}

void neuron_global_feed_forward(Neuron *neuron, double *sum, Layer &prev_layer)
{
    // Stand-in for a CUDA global function. Given a neuron, sums up the
    //  inputs from every neuron in the previous layer, as altered
    //  by the weights of the connection to that neuron. This will be used
    //  as input to the next layer of neurons.
    for (int n = 0; n < prev_layer.length; ++n) {
        *sum = *sum + prev_layer.layer[n].get_output() *
                prev_layer.layer[n].output_weights[neuron->my_index].weight;
    }

}
void neuron_global_sum_DOW(Neuron *neuron, double *sum, Layer &next_layer)
{
    // Stand-in for a CUDA global function. This is the key to backpropagation.
    //  Given a neuron, calculates the derivative of weights
    for (int n = 0; n < next_layer.length - 1; ++n) {
        *sum = *sum + neuron->output_weights[n].weight * next_layer.layer[n].gradient;
    }

}
void neuron_global_update_input_weights(Neuron *neuron, Layer &prev_layer)
{
    // Stand-in for a CUDA global function. Updates input weights based on the gradient descent
    for (int n = 0; n < prev_layer.length; ++n) {
        Neuron &prev_neuron = prev_layer.layer[n];
        double old_delta_weight = prev_neuron.output_weights[neuron->my_index].delta_weight;

        double new_delta_weight =
                // Individual input, magnified by the gradient and train rate:
                Neuron::lr
                * prev_neuron.get_output()
                * neuron->gradient
                // Also add momentum = a fraction of the previous delta weight;
                + Neuron::momentum
                * old_delta_weight;

        prev_neuron.output_weights[neuron->my_index].delta_weight = new_delta_weight;
        prev_neuron.output_weights[neuron->my_index].weight += new_delta_weight;

    }

}

void net_global_feed_forward(Layer &layer, Layer &prev_layer)
{
    // Stand-in for a CUDA global function which would run through
    //  all the layers sequentially and feed the data forward.
    for(int i=0; i < layer.length-1;++i){
        layer.layer[i].feed_forward(prev_layer);
    }

}

void net_global_update_weights(Layer &layer, Layer &prev_layer)
{
    // Stand-in for a CUDA global function which would run through
    //  all the layers sequentially and update the weights.
    for(int i=0; i < layer.length-1;++i){
        layer.layer[i].update_input_weights(prev_layer);
    }

}

void net_global_backprop(Layer &hidden_layer, Layer &next_layer)
{
    // Stand-in for a CUDA global function which would run through
    //  all the layers sequentially and calculate the gradient.
    for (int n = 0; n < hidden_layer.length; ++n) {
        hidden_layer.layer[n].calculate_hidden_gradients(next_layer);
    }

}


//--------------------------Class Functions-------------------------------------

// Unless specified otherwise, wrappers for the supposed global functions

void Neuron::update_input_weights(Layer &prev_layer)
{
    neuron_global_update_input_weights(this, prev_layer);
}

double Neuron::sum_DOW(Layer &next_layer)
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    neuron_global_sum_DOW(this, &sum, next_layer);

    return sum;
}

void Neuron::calculate_hidden_gradients(Layer &next_layer)
{
    // Calculates the hidden gradient based on the derivative of weights
    //  and the transfer function derivative
    double dow = sum_DOW(next_layer);
    gradient = dow * Neuron::transfer_function_derivative(output);
}

void Neuron::calculate_output_gradient(double target_val)
{
    // Calculates the output gradient based on error and
    //  transfer function derivative.
    double delta = target_val - output;
    gradient = delta * Neuron::transfer_function_derivative(output);
}

double Neuron::transfer_function_derivative(double x)
{
    // Derivative of tahn
    return 1.0 - x * x;
}
double Neuron::transfer_function(double x)
{
    // Transfer function used to alter the feedforward sum
    return tanh(x);
}

void Neuron::feed_forward(Layer &prev_layer)
{
    // wrapper
    double sum = 0.0;
    neuron_global_feed_forward(this, &sum, prev_layer);
    output = Neuron::transfer_function(sum);
}


Neuron::Neuron()
{
   my_index = -1;
}
 Neuron::Neuron(int num_connections, int index)
{
    // Initializes the neuron with connections ready
    //  for all neurons in the next layer.
    output_weights = new Connection[num_connections];
    for (unsigned c = 0; c < num_connections; ++c){
        output_weights[c] = Connection();
        output_weights[c].weight = Neuron::init_weight();
    }
    my_index = index;
}


Layer::Layer()
{
    length = 0;
}
Layer::Layer(int num_neurons, int num_connections)
{
    // Creates neurons and initializes them with the number
    //  of connections they will need
    layer = new Neuron[num_neurons];
    for(int i=0;i<=num_neurons;i++){
        layer[i] = Neuron(num_connections, i);
    }
    layer[num_neurons-1].set_output(1.0);
    length = num_neurons+1;
}



void Network::get_results(double *result_vals, int result_length)
{
    // Gets the final output of the output layer
    for(unsigned n = 0; n < result_length; ++n){
        Layer &output_layer = layers[NUM_HIDDEN_LAYERS+1];
        result_vals[n] = (output_layer.layer[n].get_output());
    }
}

void Network::back_prop(double * target_vals, int target_length)
{
    // Adjusts the weights to minimize output error
    Layer &output_layer = layers[NUM_HIDDEN_LAYERS+1];
    error = 0.0;
    for(unsigned n = 0; n < output_layer.length-1; ++n){
        double delta = target_vals[n] - output_layer.layer[n].get_output();
        error += delta*delta;
    }
    error /= (output_layer.length-1); //get average error squared
    error = sqrt(error); //RMS

    RAE = (RAE * RAS + error) / (RAS + 1.0);

    // Calculate output layer gradients
    for(unsigned n =0; n < output_layer.length-1; ++n){
        output_layer.layer[n].calculate_output_gradient(target_vals[n]);
    }

    // calculate gradients on hidden layers
    for(unsigned layer_num = NUM_HIDDEN_LAYERS; layer_num > 0; --layer_num){
        Layer &hidden_layer = layers[layer_num];
        Layer &next_layer = layers[layer_num+1];

        net_global_backprop(hidden_layer, next_layer);
    }

    //For all layers from outputs to first hidden layer, update connection weights
    for(unsigned layer_num = NUM_HIDDEN_LAYERS+1;layer_num > 0; --layer_num){
        Layer &layer = layers[layer_num];
        Layer &prev_layer = layers[layer_num-1];

        net_global_update_weights(layer, prev_layer);
    }

}

void Network::feed_forward(double *input_vals, int input_vals_length)
{
    // Feeds the input values into the network to get a prediction

    //assign the input values to the input neurons
    for(unsigned i = 0; i < input_vals_length; ++i){
        Layer &input_layer = layers[0];
        input_layer.layer[i].set_output(input_vals[i]);
    }

    //forward prop
    for(unsigned num_layer = 1; num_layer < NUM_HIDDEN_LAYERS+2; ++num_layer){
        Layer &layer = layers[num_layer];
        Layer &prev_layer = layers[num_layer-1];
        net_global_feed_forward(layer, prev_layer);
    }
}


Network::Network()
{
    // Network initializer, creates each layer with the number
    //  of neurons it needs.
    layers = new Layer[NUM_HIDDEN_LAYERS+2];
    layers[0] = Layer(INPUT_SIZE, HIDDEN_SIZE);
    for (int i = 1; i<NUM_HIDDEN_LAYERS; i++) {
        layers[i] = Layer(HIDDEN_SIZE, HIDDEN_SIZE);
    }
    layers[NUM_HIDDEN_LAYERS] = Layer(HIDDEN_SIZE, OUTPUT_SIZE);
    layers[1 + NUM_HIDDEN_LAYERS] = Layer(OUTPUT_SIZE, 0);

}



int main(){
    // Load the training data
    TrainingData trainData("final_training_data.txt");

    Network myNet = Network(); // create network, which initializes all internal structures

    double input_vals[INPUT_SIZE];
    double target_vals[OUTPUT_SIZE];
    double result_vals[OUTPUT_SIZE];
    int training_pass = 0;

    while (!trainData.isEof()) {
        // For every line in trainingData, feedforward and then backpropagate
        ++training_pass;
        cout << endl << "Pass " << training_pass << endl;

        // Get new input data and feed it forward:
        trainData.getNextInputs(input_vals);
        // cout << input_vals[0] << input_vals[1] << endl;

        // Get new input data and feed it forward:
        showVectorVals("Inputs:", input_vals, INPUT_SIZE);
        myNet.feed_forward(input_vals, INPUT_SIZE);

        // Collect the net's actual output results:
        myNet.get_results(result_vals, OUTPUT_SIZE);
        showVectorVals("Outputs:", result_vals, OUTPUT_SIZE);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(target_vals);
        showVectorVals("Targets:", target_vals, OUTPUT_SIZE);
        myNet.back_prop(target_vals, OUTPUT_SIZE);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: " << myNet.get_RAE() << endl;
    }
    cout << endl << "Done!" << endl;
    return 0;
}
