#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

//-----------------------Training Class to load training data-------------------

class TrainingData
{
public:
    TrainingData(const string filename);
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

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// Neuron class
class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val){m_outputVal = val;}
    double getOutputVal(void) const{return m_outputVal;}
    void feedforward(const Layer &prevLayer);
    void calculateOutputGradients(double targetVal);
    void calculateHiddenGradients(const Layer &nextlayer);
    void updateInputWeights(Layer &prevlayer);
private:
    static double eta; // overall learning rate [0.0-1.0]
    static double alpha; //multiplier of last weight change (momentum) [0.0 - n or 1]
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) {return rand()/double(RAND_MAX);}
    double sumDOW(const Layer &nextlayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevlayer){
    //the weights to be updated are in the connection container in the neurons of the preceding layer
    for(unsigned n =0 ; n < prevlayer.size();++n){
        Neuron &neuron = prevlayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        //individual weight, magnified by gradient and train rate, then add momentum
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}
double Neuron::sumDOW(const Layer &nextlayer)const{
    double sum = 0.0;
    //sum our contributions of the errors at the nodes we feed
    for(unsigned n =0; n < nextlayer.size()-1;++n){
        sum += m_outputWeights[n].weight * nextlayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calculateHiddenGradients(const Layer &nextlayer){
    double dow = sumDOW(nextlayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}
void Neuron::calculateOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunctionDerivative(double x){
    return 1.0 - x * x;
}
double Neuron::transferFunction(double x){
    ///tanh - output range [-1.0, 1.0]
    return tanh(x);
}

void Neuron::feedforward(const Layer & prevLayer){
    double sum = 0.0;

    //sum the previous layer outputs (which are our inputs)
    // include bias node from previous layer
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

class Net{
public:
    Net(const vector<unsigned> &topology);
    void feedforward(const vector<double> &inputVals);
    void backprop(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }


private:
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothing;
};

double Net::m_recentAverageSmoothing = 100.0; //Number of training samples to average over
void Net::getResults(vector<double> &resultVals) const{

    resultVals.clear();

    for(unsigned n = 0; n < m_layers.back().size()-1; ++n){

        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }

}

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

        for(unsigned n= 0; n < hiddenlayer.size();++n){
            hiddenlayer[n].calculateHiddenGradients(nextlayer);
        }

        //For all layers from outputs to first hidden layer, update connection weights
        for(unsigned layerNum = m_layers.size()-1;layerNum > 0; --layerNum){
            Layer &layer = m_layers[layerNum];
            Layer &prevlayer = m_layers[layerNum-1];

            for(unsigned n= 0; n < layer.size()-1;++n){
                layer[n].updateInputWeights(prevlayer);
            }
        }
    }
}
void Net::feedforward(const vector<double> &inputVals){

    assert(inputVals.size() == m_layers[0].size() - 1);

    //assign the input values to the input neurons
    for(unsigned i = 0; i < inputVals.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    //forward prop
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum-1];
        for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
            m_layers[layerNum][n].feedforward(prevLayer);
        }
    }



}
Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum+1];

        //We have made a new layer, now fill it with ith neurons, and add
        //a bias neuron to the layer:
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }
        //force bias to be 1.0 output
        m_layers.back().back().setOutputVal(1.0);
    }
}


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
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }
    cout << endl << "Done" << endl;
}
