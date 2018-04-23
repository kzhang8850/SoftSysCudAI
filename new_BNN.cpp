#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>


#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 32
#define INPUT_SIZE 2
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 1
#define NUM_HIDDEN_LAYERS 1

using namespace std;


//-----------------------Training Class to load training data-------------------

//TODO: make this work with arrays
class TrainingData
{
public:
    TrainingData(const string filename);
    ~TrainingData(void);
    bool isEof(void) { return m_trainingDataFile.eof(); }

    // Returns the number of input values read from the file:
    void getNextInputs(double *inputVals);
    void getTargetOutputs(double *targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}
TrainingData::~TrainingData()
{
    m_trainingDataFile.close();
}

void TrainingData::getNextInputs(double *inputVals)
{
    int index = 0;
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals[index] = oneValue;
            ++index;
        }
    }
}

void TrainingData::getTargetOutputs(double *targetOutputVals)
{
    int index = 0;
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals[index] = oneValue;
            ++index;
        }
    }

}



//----------------------------------Neural Net----------------------------------
class Connection;
class Neuron;
class Layer;
class Network;

//--------------------------------Global Functions------------------------------

void showVectorVals(string label, double *v, int length)
{
    cout << label << " ";
    for (unsigned i = 0; i < length; ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
}

void neuron_global_feed_forward(Neuron &n, int *sum, Layer &prev_layer)
{
    for (int n = 0; n < prev_layer.length; ++n) {
        *sum = *sum + prev_layer.layer[n].get_output() *
                prev_layer.layer[n].output_weights[n.my_index].weight;
    }

}
void neuron_global_sum_DOW(Neuron &n, int *sum, Layer &next_layer)
{
    for (int n = 0; n < next_layer.length - 1; ++n) {
        *sum = *sum + n.output_weights[n].weight * next_layer.layer[n].gradient;
    }

}
void neuron_global_update_input_weights(Neuron &n, Layer &prev_layer)
{
    for (int n = 0; n < prev_layer.length; ++n) {
        Neuron &neuron = prev_layer.layer[n];
        double old_delta_weight = neuron.output_weights[n.my_index].delta_weight;

        double new_delta_weight =
                // Individual input, magnified by the gradient and train rate:
                Neuron::lr
                * neuron.get_output()
                * n.gradient
                // Also add momentum = a fraction of the previous delta weight;
                + Neuron::momentum
                * old_delta_weight;

        neuron.output_weights[n.my_index].delta_weight = new_delta_weight;
        neuron.output_weights[n.my_index].weight += new_delta_weight;
    }

}



void net_global_feed_forward(Layer &layer, Layer &prev_layer)
{
    for(int i=0; i < layer.length-1;++i){
        layer.layer[i].feed_forward(prev_layer);
    }

}

void net_global_update_weights(Layer &layer, Layer &prev_layer)
{
    for(int i=0; i < layer.length-1;++i){
        layer.layer[i].update_input_weights(prev_layer);
    }

}

void net_global_backprop(Layer &hidden_layer, Layer &next_layer)
{
    for (int n = 0; n < hiddenLayer.length; ++n) {
        hidden_layer.layer[n].calculate_hidden_gradients(next_layer);
    }

}


//--------------------------Classes and Functions---------------------------------


class Connection
{
public:
    double weight;
    double delta_weight;
};

class Neuron
{
public:
    Neuron();
    void set_output(double val){output = val;}
    double get_output(void) const {return output;}
    void feed_forward();
    void calculate_output_gradient();
    void calculate_hidden_gradients();
    void update_input_weights();
private:
    static double lr;
    static double momentum;
    static double transfer_function(double x);
    static double transfer_function_derivative(double x);
    static double init_weight(void) {return rand()/double(RAND_MAX);}
    double sum_DOW() const;
    double output;
    Connection* output_weights;
    unsigned my_index;
    double gradient;

};
double Neuron::lr = 0.15;
double Neuron::momentum = 0.5;

void Neuron::update_input_weights(Layer &prev_layer)
{
    neuron_global_update_input_weights(this, prev_layer);
}


double Neuron::sum_DOW(Layer &next_layer)const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    neuron_global_sum_DOW(this, &sum, next_layer);

    return sum;
}

void Neuron::calculate_hidden_gradients(Layer &next_layer)
{
    double dow = sum_DOW(next_layer);
    gradient = dow * Neuron::transfer_function_derivative(output);
}



void Neuron::calculate_output_gradient(double target_val)
{
    double delta = target_val - output;
    gradient = delta * Neuron::transfer_function_derivative(output);
}


double Neuron::transfer_function_derivative(double x)
{
    return 1.0 - x * x;
}
double Neuron::transfer_function(double x)
{
    ///tanh - output range [-1.0, 1.0]
    return tanh(x);
}



void Neuron::feed_forward(Layer &prev_layer)
{
    double sum = 0.0;
    neuron_global_feed_forward(this, &sum, prev_layer);
    output = Neuron::transfer_function(sum);
}


 Neuron::Neuron(int num_connections, int index)
{
    output_weights = new Connection[num_connections];
    for (unsigned c = 0; c < num_connections; ++c){
        output_weights[i] = Connection();
        output_weights[i].weight = randomWeight();
    }
    my_index = index;
}

class Layer
{
public:
    Layer(int num_neurons, int num_connections);
    Neuron* layer;
    int length;
};

Layer::Layer(int num_neurons, int num_connections)
{
    layer = new Neuron[num_neurons];
    for(int i=0;i<=num_neurons;i++){
        layer[i] = Neuron(num_connections, i);
    }
    layer[num_neurons-1].set_output(1.0);
    length = num_neurons+1;
}

class Network
{
public:
    Network();
    void feed_forward(double *input_vals, int input_vals_length);
    void back_prop();

    void get_results(double * result_vals, int result_length) const;
    double get_RAE() const { return RAE; }

private:
    Layer *layers;
    double error;
    double RAE;
    static double RAS;

};
double Network::RAS = 100.0; //Number of training samples to average over


void Network::get_results(double * result_vals, int result_length) const
{
    for(unsigned n = 0; n < result_length; ++n){
        Layer &output_layer = layers[NUM_HIDDEN_LAYERS+1];
        result_vals[n] = (output_layer.layer[n].get_output());
    }
}

void Network::back_prop(double * target_vals, int target_length)
{
    Layer &outputLayer = layers[NUM_HIDDEN_LAYERS+1];
    error = 0.0;
    for(unsigned n = 0; n < OUTPUT_SIZE; ++n){
        double delta = targetVals[n] - outputLayer.layer[n].get_output();
        error += delta*delta;
    }
    error /= OUTPUT_SIZE; //get average error squared
    error = sqrt(error); //RMS

    RAE = (RAE * RAS + error) / (RAS + 1.0);

    // Calculate output layer gradients
    for(unsigned n =0; n < OUTPUT_SIZE; ++n){
        outputLayer.layer[n].calculate_output_gradient(target_vals[n]);
    }

    // calculate gradients on hidden layers
    for(unsigned layer_num = NUM_HIDDEN_LAYERS; layer_num > 0; --layer_num){
        Layer &hiddenlayer = layers[layer_num];
        Layer &nextlayer = layers[layer_num+1];

        net_global_backprop(hiddenlayer, nextlayer);
    }

    //For all layers from outputs to first hidden layer, update connection weights
    for(unsigned layer_num = NUM_HIDDEN_LAYERS+1;layer_num > 0; --layer_num){
        Layer &layer = layers[layer_num];
        Layer &prevlayer = layers[layer_num-1];

        net_global_update_weights(layer, prevLayer);
    }

}

void Network::feed_forward(double *input_vals, int input_vals_length)
{
    //assign the input values to the input neurons
    for(unsigned i = 0; i < input_vals_length; ++i){
        input_layer = layers[0]
        input_layer.layer[i].setOutputVal(input_vals[i]);
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
    layers = new Layer[NUM_HIDDEN_LAYERS+2];
    layers[0] = Layer(INPUT_SIZE, HIDDEN_SIZE);
    for (int i = 1; i<NUM_HIDDEN_LAYERS; i++) {
        layers[i] = Layer(HIDDEN_SIZE, HIDDEN_SIZE);
    }
    layers[NUM_HIDDEN_LAYERS] = Layer(HIDDEN_SIZE, OUTPUT_SIZE);
    layers[1 + NUM_HIDDEN_LAYERS] = Layer(OUTPUT_SIZE, 0);

}



int main(){
    TrainingData trainData("trainingdata.txt");

    Network myNet = Network();

    //TODO: change these into arrays
    double input_vals[INPUT_SIZE];
    double target_vals[OUTPUT_SIZE];
    double result_vals[OUTPUT_SIZE];
    int training_pass = 0;

    while (!trainData.isEof()) {
        ++training_pass;
        cout << endl << "Pass " << training_pass;

        // Get new input data and feed it forward:
        trainData.getNextInputs(input_vals);

        // Get new input data and feed it forward:
        showVectorVals(": Inputs:", input_vals, INPUT_SIZE);
        // myNet.feed_forward(input_vals, INPUT_SIZE);

        // Collect the net's actual output results:
        // myNet.get_results(result_vals, OUTPUT_SIZE);
        showVectorVals("Outputs:", result_vals, OUTPUT_SIZE);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(target_vals);
        showVectorVals("Targets:", target_vals, OUTPUT_SIZE);
        // myNet.back_prop(target_vals, OUTPUT_SIZE);

        // Report how well the training is working, average over recent samples:
        // cout << "Net recent average error: " << myNet.get_RAE() << endl;
    }
    cout << endl << "Done!" << endl;
    return 0;
}
