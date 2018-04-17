#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 32
#define INPUT_SIZE 2
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 1
#define NUM_HIDDEN_LAYERS 1

using namespace std;

//-----Unified Memory Class Constructor; all shared memory classes inherit------

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


//-----------------------Training Class to load training data-------------------

class TrainingData : public Managed
{
public:
    TrainingData(const string filename);
    ~TrainingData(void);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}
TrainingData::~TrainingData()
{
    m_trainingDataFile.close();
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

//----------------------------------Neural Net----------------------------------
class Connection;
class Neuron;
class Layer;
class Net;

class Connection: public Managed{
    double weight;
    double deltaWeight;
};

class Neuron : public Managed {
public:
    __host__ __device__ void setOutputVal(double val){m_outputVal = val;}
    __host__ __device__ double getOutputVal(void) const{return m_outputVal;}
    __device__ void feedforward(const Layer &prevLayer);
    __device__ void calculateOutputGradients(double targetVal);
    __device__ void calculateHiddenGradients(const Layer &nextlayer);
    void updateInputWeights(Layer &prevlayer);
private:
    static double eta; // overall learning rate [0.0-1.0]
    static double alpha; //multiplier of last weight change (momentum) [0.0 - n or 1]
    __device__ static double transferFunction(double x);
    __device__ static double transferFunctionDerivative(double x);
    __host__ static double randomWeight(void) {return rand()/double(RAND_MAX);}
    __device__ double sumDOW(const Layer &nextlayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};


class Layer: public Managed{
    Neuron* neurons;        //The array of neurons
};


class Net : public Managed {
public:
    Net(const vector<unsigned> &topology);
    void feedforward(const vector<double> &inputVals);
    void backprop(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }


private:
    Layer* layers[NUM_HIDDEN_LAYERS + 2]; //m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothing;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;
double Net::m_recentAverageSmoothing = 100.0; //Number of training samples to average over


//This function updates the weights of each neuron in the layer
//prevLayer is the layer to be updated
__global__ 
void d_updateInputWeights(Neuron &neuron, Layer &prevLayer){
    for(unsigned n = blockIdx.x*blockDim.x +  threadIdx.x;
        n < prevLayer.size();
        n += blockDim.x * gridDim.x)
    {
        Neuron &prev_neuron = prevlayer[n];
        double oldDeltaWeight = prev_neuron.m_outputWeights[neuron.m_myIndex].deltaWeight;
        //individual weight, magnified by gradient and train rate, then add momentum
        double newDeltaWeight = neuron.eta * prev_neuron.getOutputVal() * neuron.m_gradient + neuron.alpha * oldDeltaWeight;

        prev_neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        prev_neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}


__global__ 
void d_sumDOW(double *sum, Layer &nextLayer){
    for(unsigned n = blockIdx.x*blockDim.x +  threadIdx.x;
        n < nextLayer.size()-1;
        n += blockDim.x * gridDim.x)
    {
        *sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
}


__global__ 
void calculateHiddenGradients(Neuron &neuron, const Layer &nextlayer){
    double dow = neuron.sumDOW(nextlayer);
    neuron.m_gradient = dow * Neuron::transferFunctionDerivative(neuron.m_outputVal);
}


__global__ 
void d_feedForward(Neuron &neuron, double *sum, Layer &prevLayer){
    for(unsigned n = blockIdx.x*blockDim.x +  threadIdx.x;
        n < prevLayer.size();
        n += blockDim.x * gridDim.x)
    {
        *sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[neuron.m_myIndex].weight;
    }
}


__device__ 
void Neuron::updateInputWeights(Layer &prevlayer){
    //the weights to be updated are in the connection container in the neurons of the preceding layer
    d_updateInputWeights<<<NUM_BLOCKS,THREADS_PER_BLOCK>>> (this, prevlayer);
    cudaDeviceSynchronize();
}


__device__ 
double Neuron::sumDOW(const Layer &nextlayer)const{
    double sum = 0.0;
    //sum our contributions of the errors at the nodes we feed
    d_sumDOW<<<NUM_BLOCKS,THREADS_PER_BLOCK>>> (this, &sum, nextlayer);
    cudaDeviceSynchronize();
    return sum;
}


__device__ 
void Neuron::calculateOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

__device__ 
double Neuron::transferFunctionDerivative(double x){
    return 1.0 - x * x;
}
__device__ 
double Neuron::transferFunction(double x){
    ///tanh - output range [-1.0, 1.0]
    return tanh(x);
}


__device__ 
void Neuron::feedforward(const Layer & prevLayer){
    double sum = 0.0;

    //sum the previous layer outputs (which are our inputs)
    // include bias node from previous layer
    d_feedForward<<<NUM_BLOCKS,THREADS_PER_BLOCK>>> (this, &sum, prevLayer);
    cudaDeviceSynchronize();
    m_outputVal = Neuron::transferFunction(sum);
}

__host__
__device__
 Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}


__host__
__device__
 Layer::Layer(unsigned layer_size, unsigned num_connections) {
    neurons = new Neuron*[layer_size];
    for (int i = 0; i<layer_size; i++) {
        neurons[i] = 
    }
 }


__device__ 
void Net::getResults(vector<double> &resultVals) const{

    resultVals.clear();

    for(unsigned n = 0; n < m_layers.back().size()-1; ++n){

        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }

}

__global__ 
void Net::hidden_backprop(Layer &hiddenLayer, Layer &nextLayer){
    calculateHiddenGradients <<<Blah, blah>>> (args);
    // for(unsigned n = blockIdx.x*blockDim.x +  threadIdx.x;
    //     n < hiddenLayer.size();
    //     n += blockDim.x * gridDim.x)
    // {
    //     hiddenlayer[n].calculateHiddenGradients(nextlayer);
    // }
}
__global__ 
void Net::update_weights(Layer layer, Layer prevLayer){
    for(unsigned n = blockIdx.x*blockDim.x +  threadIdx.x;
        n < layer.size()-1;
        n += blockDim.x * gridDim.x)
    {
        layer[n].updateInputWeights(prevLayer);
    }
}

__host__
void Net::backprop(const vector<double> &targetVals){
    // calculate overall net error (RMS of outputs neuron errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for(unsigned n = 0; n < outputLayer.size()-1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta*delta;
    }
    m_error /= outputLayer.size()-1; //get average error squared
    m_error = sqrt(m_error); //RMS

    //Implement a recent average measurement

    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothing + m_error)
                            / (m_recentAverageSmoothing + 1.0);

    // Calculate output layer gradients
    for(unsigned n =0; n < outputLayer.size()-1; ++n){
        outputLayer[n].calculateOutputGradients(targetVals[n]);
    }

    // calculate gradients on hidden layers
    for(unsigned layerNum = m_layers.size()-2; layerNum > 0; --layerNum){
        Layer &hiddenlayer = m_layers[layerNum];
        Layer &nextlayer = m_layers[layerNum+1];

        Net::hidden_backprop<<<1, 32>>> (hiddenlayer, nextlayer);
        cudaDeviceSynchronize();

        //For all layers from outputs to first hidden layer, update connection weights
        for(unsigned layerNum = m_layers.size()-1;layerNum > 0; --layerNum){
            Layer &layer = m_layers[layerNum];
            Layer &prevlayer = m_layers[layerNum-1];

            Net::update_weights<<<NUM_BLOCKS,THREADS_PER_BLOCK>>> (layer, prevLayer);
            cudaDeviceSynchronize();
        }
    }
}

__global__ 
void Net::d_feedForward(Layer() &prevLayer, unsigned layerNum){
    for(unsigned n = blockIdx.x*blockDim.x +  threadIdx.x;
        n < m_layers[layerNum].size() - 1;
        n += blockDim.x * gridDim.x)
    {
        m_layers[layerNum][n].feedforward(prevLayer);
    }
}


__host__ 
void Net::feedforward(const vector<double> &inputVals){

    assert(inputVals.size() == m_layers[0].size() - 1);

    //assign the input values to the input neurons
    for(unsigned i = 0; i < inputVals.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    //forward prop
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum-1];
        Net::d_feedForward<<<1, 32>>> (prevLayer, layerNum);
        cudaDeviceSynchronize();
    }
}


__host__
__device__
 Net::Net(const vector<unsigned> &topology){
    layers[0] = new Layer(INPUT_SIZE, HIDDEN_SIZE);
    for (int i = 1; i<HIDDEN_SIZE; i++) {
        layers[i] = new Layer(HIDDEN_SIZE, HIDDEN_SIZE);
    }
    layers[HIDDEN_SIZE] = new Layer(HIDDEN_SIZE, OUTPUT_SIZE);
    layers[1 + HIDDEN_SIZE] = new Layer(OUTPUT_SIZE, 0);

}


__device__
void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
}



int main(){
    TrainingData trainData("trainingdata.txt");

    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedforward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());
        myNet.backprop(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
    }
    cout << endl << "Done" << endl;
    return 0
}
