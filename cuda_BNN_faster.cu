#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include "training_reader.cuh"


#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 4
#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define NUM_HIDDEN_LAYERS 1
#define MOMENTUM 0.5
#define LR 0.15
#define TRAINING_SIZE 100001

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



//----------------------------------Neural Declarations-------------------------
class Connection;
class Neuron;
class Layer;
class Network;

//--------------------------------Global Declarations---------------------------

__host__ void showVectorVals(string label, double *v, int length);
__global__ void neuron_global_feed_forward(Neuron *n, double *sum, Layer &prev_layer);
__global__ void neuron_global_sum_DOW(Neuron *n, double *sum, Layer &next_layer);
__global__ void neuron_global_update_input_weights(Neuron *n, Layer &prev_layer);
__global__ void net_global_feed_forward(Layer &layer, Layer &prev_layer);
__global__ void net_global_update_weights(Layer &layer, Layer &prev_layer);
__global__ void net_global_backprop(Layer &hidden_layer, Layer &next_layer);


//-------------------------------Net Class Initializations----------------------

class Connection: public Managed
{
public:
    double weight;
    double delta_weight;
};


class Neuron: public Managed
{
public:
    __host__ Neuron();
    // __host__ ~Neuron();
    __host__ Neuron(int num_neurons, int num_connections);
    __host__ __device__ void set_output(double val){output = val;}
    __host__ __device__ double get_output(void) {return output;}
    __host__ __device__ void feed_forward(Layer &prev_layer);
    __host__ __device__ void calculate_output_gradient(double target_val);
    __host__ __device__ void calculate_hidden_gradients(Layer &next_layer);
    __host__ __device__ void update_input_weights(Layer &prev_layer);
    double output;
    Connection* output_weights;
    unsigned my_index;
    double gradient;
    double* DOW_sum;
    double* FF_sum;

private:

    __host__ __device__ static double transfer_function(double x);
    __host__ __device__ static double transfer_function_derivative(double x);
    static double init_weight(void) {return rand()/double(RAND_MAX);}
    __host__ __device__ double sum_DOW(Layer &next_layer);


};



class Layer: public Managed
{
public:
    __host__ Layer();
    __host__ Layer(int num_neurons, int num_connections);
    Neuron* layer;
    int length;
};


class Network: public Managed
{
public:
    __host__ Network();
    // __host__ ~Network();
    __host__ __device__ void feed_forward(double *input_vals, int input_vals_length);
    __host__ __device__ void back_prop(double * target_vals, int target_length);

    __host__ __device__ void get_results(double *result_vals, int result_length);
    __host__ __device__ double get_RAE() const { return RAE; }

private:
    Layer *layers;
    double error;
    double RAE;
    double RAS;

};


//------------------------------Global Functions--------------------------------

// __host__
// __device__
// void showVectorVals(string label, double *v, int length)
// {
//     cout << label << " ";
//     for (unsigned i = 0; i < length; ++i) {
//         cout << v[i] << " ";
//     }
//     cout << endl;
// }

__global__
void neuron_global_feed_forward(Neuron *neuron, double *sum, Layer &prev_layer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Neuron &neuron = *n;
    for (int n = index; n < prev_layer.length; n+=stride) {
        *sum = *sum + prev_layer.layer[n].get_output() *
                prev_layer.layer[n].output_weights[neuron->my_index].weight;
    }

}

__global__
void neuron_global_sum_DOW(Neuron *neuron, double *sum, Layer &next_layer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Neuron &neuron = *n;
    for (int n = index; n < next_layer.length - 1; n+=stride) {
        *sum = *sum + neuron->output_weights[n].weight * next_layer.layer[n].gradient;
    }

}
__global__
void neuron_global_update_input_weights(Neuron *neuron, Layer &prev_layer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Neuron &neuron = *n;
    for (int n = index; n < prev_layer.length; n+=stride) {
        Neuron &prev_neuron = prev_layer.layer[n];
        double old_delta_weight = prev_neuron.output_weights[neuron->my_index].delta_weight;

        double new_delta_weight =
                // Individual input, magnified by the gradient and train rate:
                LR
                * prev_neuron.get_output()
                * neuron->gradient
                // Also add momentum = a fraction of the previous delta weight;
                + MOMENTUM
                * old_delta_weight;

        prev_neuron.output_weights[neuron->my_index].delta_weight = new_delta_weight;
        // cout << "DELTA WEIGHT " << new_delta_weight << endl;
        prev_neuron.output_weights[neuron->my_index].weight += new_delta_weight;

        // cout << "WEIGHT: " << prev_neuron.output_weights[neuron.my_index].weight << endl;
    }

}

__global__
void net_global_feed_forward(Layer &layer, Layer &prev_layer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index; i < layer.length-1;i +=stride){
        layer.layer[i].feed_forward(prev_layer);
    }

}

__global__
void net_global_update_weights(Layer &layer, Layer &prev_layer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index; i < layer.length-1;i+=stride){
        layer.layer[i].update_input_weights(prev_layer);
    }

}

__global__
void net_global_backprop(Layer &hidden_layer, Layer &next_layer)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n = index; n < hidden_layer.length; n+=stride) {
        hidden_layer.layer[n].calculate_hidden_gradients(next_layer);
    }

}

__global__
void global_training(Network *network, double* inputs, double* targets, double* errors, size_t input_pitch, size_t target_pitch, size_t errors_pitch)
{
    Network net = *network;
    for(int i = 0; i < TRAINING_SIZE; ++i){
        net.feed_forward(inputs+i*input_pitch, INPUT_SIZE);
        net.get_results(errors+i*errors_pitch, OUTPUT_SIZE);
        net.back_prop(targets+i*target_pitch, OUTPUT_SIZE);
    }

}


//--------------------------Class Functions-------------------------------------
__host__
__device__
void Neuron::update_input_weights(Layer &prev_layer)
{

    neuron_global_update_input_weights<<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (this, prev_layer);
    cudaDeviceSynchronize();
}
__host__
__device__
double Neuron::sum_DOW(Layer &next_layer)
{
    // double* sum;
    *DOW_sum = 0.0;
    // Sum our contributions of the errors at the nodes we feed.

    neuron_global_sum_DOW<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(this, DOW_sum, next_layer);
    cudaDeviceSynchronize();
    return *DOW_sum;
}
__host__
__device__
void Neuron::calculate_hidden_gradients(Layer &next_layer)
{
    double dow = sum_DOW(next_layer);
    gradient = dow * Neuron::transfer_function_derivative(output);
}

__host__
__device__
void Neuron::calculate_output_gradient(double target_val)
{
    double delta = target_val - output;
    gradient = delta * Neuron::transfer_function_derivative(output);
}

__host__
__device__
double Neuron::transfer_function_derivative(double x)
{
    return 1.0 - x * x;
}
__host__
__device__
double Neuron::transfer_function(double x)
{
    ///tanh - output range [-1.0, 1.0]
    return tanh(x);
}
__host__
__device__
void Neuron::feed_forward(Layer &prev_layer)
{
    // double* sum;
    // malloc(&sum, sizeof(double));
    // *sum = 0.0;
    *FF_sum = 0.0;

    neuron_global_feed_forward<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(this, FF_sum, prev_layer);
    cudaDeviceSynchronize();
    output = Neuron::transfer_function(*FF_sum);
}

__host__
Neuron::Neuron()
{
   my_index = 999;
}
__host__
 Neuron::Neuron(int num_connections, int index)
{
    cudaMallocManaged(&output_weights, sizeof(Connection)*num_connections);
    for (unsigned c = 0; c < num_connections; ++c){
        output_weights[c] = Connection();
        output_weights[c].weight = Neuron::init_weight();
    }
    cudaMallocManaged(&DOW_sum, sizeof(double));
    cudaMallocManaged(&FF_sum, sizeof(double));
    *DOW_sum = 0.0;
    *FF_sum = 0.0;
    my_index = index;
}
// __host__
// Neuron::~Neuron()
// {
//     cudaFree(DOW_sum);
//     cudaFree(FF_sum);
// }

__host__
Layer::Layer()
{
    length = 0;
}
__host__
Layer::Layer(int num_neurons, int num_connections)
{
    cudaMallocManaged(&layer, sizeof(Neuron)*num_neurons);
    for(int i=0;i<=num_neurons;i++){
        layer[i] = Neuron(num_connections, i);
    }
    layer[num_neurons-1].set_output(1.0);
    length = num_neurons+1;
}


__host__
__device__
void Network::get_results(double *result_vals, int result_length)
{
    for(unsigned n = 0; n < result_length; ++n){
        Layer &output_layer = layers[NUM_HIDDEN_LAYERS+1];
        result_vals[n] = (output_layer.layer[n].get_output());
    }
}

__host__
__device__
void Network::back_prop(double * target_vals, int target_length)
{
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

        net_global_backprop<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(hidden_layer, next_layer);
        cudaDeviceSynchronize();
    }

    //For all layers from outputs to first hidden layer, update connection weights
    for(unsigned layer_num = NUM_HIDDEN_LAYERS+1;layer_num > 0; --layer_num){
        Layer &layer = layers[layer_num];
        Layer &prev_layer = layers[layer_num-1];

        net_global_update_weights<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(layer, prev_layer);
        cudaDeviceSynchronize();
    }

}

__host__
__device__
void Network::feed_forward(double *input_vals, int input_vals_length)
{
    //assign the input values to the input neurons
    for(unsigned i = 0; i < input_vals_length; ++i){
        Layer &input_layer = layers[0];
        input_layer.layer[i].set_output(input_vals[i]);
    }

    //forward prop
    for(unsigned num_layer = 1; num_layer < NUM_HIDDEN_LAYERS+2; ++num_layer){
        Layer &layer = layers[num_layer];
        Layer &prev_layer = layers[num_layer-1];
        net_global_feed_forward<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(layer, prev_layer);
        cudaDeviceSynchronize();
    }
}


__host__
Network::Network()
{
    cudaMallocManaged(&layers, sizeof(Layer)*(NUM_HIDDEN_LAYERS+2));
    layers[0] = Layer(INPUT_SIZE, HIDDEN_SIZE);
    for (int i = 1; i<NUM_HIDDEN_LAYERS; i++) {
        layers[i] = Layer(HIDDEN_SIZE, HIDDEN_SIZE);
    }
    layers[1] = Layer(HIDDEN_SIZE, OUTPUT_SIZE);
    layers[1 + NUM_HIDDEN_LAYERS] = Layer(OUTPUT_SIZE, 0);
    RAS = 100.0; //Number of training samples to average over


}
// __host__
// Network::~Network()
// {
//     cudaFree(layers);
// }



int main(){
    TrainingData trainData("/home/rooster/Documents/SoftSysCudAI/faster_training_data.txt");
    cout << " I got the data file" << endl;
    Network myNet = Network();

    double temp_inputs[INPUT_SIZE];
    double temp_targets[OUTPUT_SIZE];

    double** input_array = (double **) malloc(TRAINING_SIZE * sizeof(double *));
    for(int i=0;i<TRAINING_SIZE;++i){
        input_array[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
    }
    double** target_array = (double **) malloc(TRAINING_SIZE * sizeof(double *));
    for(int i=0;i<TRAINING_SIZE;++i){
        target_array[i] = (double *)malloc(OUTPUT_SIZE * sizeof(double));
    }
    double** result_array = (double **) malloc(TRAINING_SIZE * sizeof(double *));
    for(int i=0;i<TRAINING_SIZE;++i){
        result_array[i] = (double *)malloc(OUTPUT_SIZE * sizeof(double));
    }

    cout << " I made the mallocs" << endl;

    // cudaMalloc(&input_array, sizeof(double)*TRAINING_SIZE);
    // cudaMalloc(&target_array, sizeof(double)*TRAINING_SIZE);
    // cudaMalloc(&result_array, sizeof(double)*TRAINING_SIZE);
    int index = 0;

    while (!trainData.isEof()) {
        // ++training_pass;
        // cout << endl << "Pass " << training_pass;

        // Get new input data and feed it forward:
        trainData.getNextInputs(temp_inputs);
        for(int i=0;i<INPUT_SIZE;i++){
            input_array[index][i] = temp_inputs[i];
        }

        // Get new input data and feed it forward:
        // showVectorVals(": Inputs:", input_vals, INPUT_SIZE);
        // myNet.feed_forward(input_vals, INPUT_SIZE);

        // Collect the net's actual output results:
        // myNet.get_results(result_vals, OUTPUT_SIZE);
        // showVectorVals("Outputs:", result_vals, OUTPUT_SIZE);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(temp_targets);
        for(int i=0;i<OUTPUT_SIZE;i++){
            target_array[index][i] = temp_targets[i];
        }
        index++;
        // cout<<index<<endl;
        // showVectorVals("Targets:", target_vals, OUTPUT_SIZE);
        // myNet.back_prop(target_vals, OUTPUT_SIZE);

        // Report how well the training is working, average over recent samples:
        // cout << "Net recent average error: " << myNet.get_RAE() << endl;
    }
    cout << " I read the data file" << endl;
    size_t input_pitch;
    size_t target_pitch;
    size_t errors_pitch;
    double *inputs;
    double *targets;
    double *errors;
    cudaMallocPitch(&inputs, &input_pitch, sizeof(double)*INPUT_SIZE, TRAINING_SIZE);
    cudaMallocPitch(&targets, &target_pitch, sizeof(double)*OUTPUT_SIZE, TRAINING_SIZE);
    cudaMallocPitch(&errors, &errors_pitch, sizeof(double)*OUTPUT_SIZE, TRAINING_SIZE);
    cout << " I got cuda mallocs" << endl;
    cudaMemcpy2D(inputs, input_pitch, input_array, INPUT_SIZE*sizeof(double), INPUT_SIZE*sizeof(double), TRAINING_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy2D(targets, target_pitch, target_array, OUTPUT_SIZE*sizeof(double), OUTPUT_SIZE*sizeof(double), TRAINING_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy2D(errors, errors_pitch, result_array, OUTPUT_SIZE*sizeof(double), OUTPUT_SIZE*sizeof(double), TRAINING_SIZE, cudaMemcpyHostToDevice);
    cout << " I copied" << endl;

    for(int i = 0; i <TRAINING_SIZE; ++i){
        for(int j=0; j<INPUT_SIZE; ++j){
            double * temp = (double *)(((char *)inputs)+(j*input_pitch));
            cout << "hello" << endl;
            cout << temp[i] << endl;

        }
    }

    // global_training<<<1,1>>>(&myNet, inputs, targets, errors, input_pitch, target_pitch, errors_pitch);

    cout << " I done running" << endl;
    cudaMemcpy2D(result_array, errors_pitch, errors, OUTPUT_SIZE*sizeof(double), OUTPUT_SIZE*sizeof(double), TRAINING_SIZE, cudaMemcpyDeviceToHost);

    for(int i=0;i<TRAINING_SIZE;++i){
        cout << "Error: " ;
        cout << *(result_array+i*errors_pitch) << " ";

        cout << endl;

    }


    cudaFree(inputs);
    cudaFree(targets);
    cudaFree(errors);

    for(int i=0;i<TRAINING_SIZE;++i){
        free(input_array[i]);
    }
    for(int i=0;i<TRAINING_SIZE;++i){
        free(target_array[i]);
    }
    for(int i=0;i<TRAINING_SIZE;++i){
        free(result_array[i]);
    }
    free(input_array);
    free(target_array);
    free(result_array);

    cout << endl << "Done!" << endl;
    return 0;
}
