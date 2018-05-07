#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <curand.h>
#include <curand_kernel.h>
#include "training_reader.cuh"


#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 4
#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define NUM_HIDDEN_LAYERS 1
#define MOMENTUM 0.5
#define LR 0.15
#define TRAINING_SIZE 100002
#define BLOCKSIZE_x 2
#define BLOCKSIZE_y 2

using namespace std;


//----------------------------------Neural Declarations-------------------------

class Connection;
class Neuron;
class Layer;
class Network;

//--------------------------------Global Declarations---------------------------

__global__ void neuron_global_feed_forward(Neuron *n, double *sum, Layer &prev_layer);
__global__ void neuron_global_sum_DOW(Neuron *n, double *sum, Layer &next_layer);
__global__ void neuron_global_update_input_weights(Neuron *n, Layer &prev_layer);
__global__ void net_global_feed_forward(Layer &layer, Layer &prev_layer);
__global__ void net_global_update_weights(Layer &layer, Layer &prev_layer);
__global__ void net_global_backprop(Layer &hidden_layer, Layer &next_layer);


//-------------------------------Net Class Initializations----------------------

class Connection
{
    // weighted connection between two neurons on different layers.
    //  delta_weight (how much it should change) is calculated during
    //  backpropagation, based on the output error.
public:
    double weight;
    double delta_weight;
};


class Neuron
{
    // Perceptron node that outputs the sum of all inputs times the weight
    //  of the connection between it and the neuron before, then passes
    //  this value through the connections to all neurons in the next layer.
public:
    __host__ __device__ Neuron();
    __host__ __device__ ~Neuron(){};
    __host__ __device__ Neuron(int num_neurons, int num_connections);
    __host__ __device__ void set_output(double val){output = val;}
    __host__ __device__ double get_output(void) {return output;}
    __host__ __device__ void feed_forward(Layer *prev_layer);
    __host__ __device__ void calculate_output_gradient(double target_val);
    __host__ __device__ void calculate_hidden_gradients(Layer *next_layer);
    __host__ __device__ void update_input_weights(Layer *prev_layer);
    double output;
    Connection* output_weights;
    unsigned my_index;
    double gradient;
    double* DOW_sum;
    double* FF_sum;

private:

    __host__ __device__ static double transfer_function(double x);
    __host__ __device__ static double transfer_function_derivative(double x);
    // __host__ __device__ double init_weight(double *result); // rand does not work in global functions
    __host__ __device__ double sum_DOW(Layer *next_layer);


};



class Layer
{
    // Container for all the neurons in a layer. Acts as an array.
public:
    __host__ __device__ Layer();
    __host__ __device__ ~Layer(){};
    __host__ __device__ Layer(int num_neurons, int num_connections);
    Neuron* layer;
    int length;
};


class Network
{
    // Container for all layers and wrapper for calls to individual neurons.
public:
    __host__ __device__ Network();
    __host__ __device__ ~Network(){};
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

__global__
void init_weight(double *result) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
    // curandState_t state;

    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock(), 0, 0, &state);

    *result = curand_uniform_double(&state);

    //   /* curand works like rand - except that it takes a state as a parameter */
    //   *result = recurand(&state) % MAX;
}

__global__
void neuron_global_feed_forward(Neuron *neuron, double *sum, Layer *pl)
{
    // Stands in as a neuron's sum. Gets all the outputs from the previous
    //  layer, multiplies them by the weights of the connections, and sums
    //  the results. The sum is assigned to the output of the given neuron.

    // In theory, this should be parallelizable for all neurons within the same
    //  layer, as neurons on the same layer do not effect each other. This did
    //  not seem to be the case.
    Layer prev_layer = *pl;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n = index; n < prev_layer.length; n+=stride) {
        *sum = *sum + prev_layer.layer[n].get_output() *
                prev_layer.layer[n].output_weights[neuron->my_index].weight;
    }

}

__global__
void neuron_global_sum_DOW(Neuron *neuron, double *sum, Layer *nl)
{
    // Sums the Derivative of Weights, which will be used to calculate the
    //  gradient and adjust the weights of the connections for the next pass.

    // Should be parallelizable within a layer.
    Layer next_layer = *nl;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n = index; n < next_layer.length - 1; n+=stride) {
        *sum = *sum + neuron->output_weights[n].weight * next_layer.layer[n].gradient;
    }

}
__global__
void neuron_global_update_input_weights(Neuron *neuron, Layer *pl)
{
    // Based on the previous delta_weight and the gradient, updates the input
    //  weight to each neuron in order to minimize error from output.
    Layer prev_layer = *pl;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n = index; n < prev_layer.length; n+=stride) {
        Neuron prev_neuron = prev_layer.layer[n];
        double old_delta_weight = prev_neuron.output_weights[neuron->my_index].delta_weight;

        double new_delta_weight =
                // Individual input, magnified by the gradient and train rate:
                LR
                * prev_neuron.get_output()
                * neuron->gradient
                // Also add momentum = a fraction of the previous delta weight;
                + MOMENTUM
                * old_delta_weight;

        // Adjust the connections
        prev_neuron.output_weights[neuron->my_index].delta_weight = new_delta_weight;
        prev_neuron.output_weights[neuron->my_index].weight += new_delta_weight;
    }

}

__global__
void net_global_feed_forward(Layer *l, Layer *pl)
{
    // Wrapper around each step of the feed forward process. Given two sequential
    //  layers, iterates through the neurons and performs their feed_forward method.
    //  Each layer must run sequentially, and thus is not parallelizable.
    Layer layer = *l;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index; i < layer.length-1;i +=stride){
        // Call to the neuron device function feed_forward.
        //  Ideally, all neuron feed_forwards are calculated simultaneously.
        layer.layer[i].feed_forward(pl);
    }

}

__global__
void net_global_update_weights(Layer *l, Layer *pl)
{
    // Wrapper around the update weight process. Given two sequential layers,
    //  iterates through the neurons and performs update_input_weights.
    //  Each layer must run sequentially, and thus is not parallelizable.
    Layer layer = *l;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index; i < layer.length-1;i+=stride){
        // Call to the neuron device function update_input_weights.
        //  Ideally, all neuron update_input_weights are calculated simultaneously.
        layer.layer[i].update_input_weights(pl);
    }

}

__global__
void net_global_backprop(Layer *hl, Layer *nl)
{
    // Wrapper around the backpropagation. Given two sequential layers,
    //  iterates through the neurons and calculates the gradients.
    //  Each layer must run sequentially, and thus is not parallelizable.
    Layer hidden_layer = *hl;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n = index; n < hidden_layer.length; n+=stride) {
        // Call to the neuron device function calculate_hidden_gradients.
        //  Ideally, all neuron calculate_hidden_gradients are calculated simultaneously
        hidden_layer.layer[n].calculate_hidden_gradients(nl);
    }

}

__global__
void global_training(Network *net, double* inputs, double* targets, double* errors, size_t input_pitch, size_t target_pitch, size_t errors_pitch)
{
    // Initializes and runs all the other functions needed to train our network. Most of our issues
    //  arise from here.
    Network network = *net; // Pointer to the cudaMalloc'ed space for the network. Currently, empty space.
    network = Network(); // Calls the network initializer to create all layers, neurons, and connections in GPU memory.
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    while (tidy < TRAINING_SIZE)
    {
        // For every row of training data we have, get the inputs, expected outputs, and create space for errors.
       double *row_a = (double *)((char*)inputs + tidy * input_pitch);
       network.feed_forward(row_a, INPUT_SIZE);
       double *row_b = (double *)((char*)errors + tidy * errors_pitch);
       double *row_c = (double *)((char*)targets + tidy * target_pitch);

       network.get_results(row_b, OUTPUT_SIZE); // gets the results and assigns them to row_b
       network.back_prop(row_c, OUTPUT_SIZE); // based on the target value, updates the weights

       printf("Error: %f", network.get_RAE()); // prints the calculated error

       tidx = (tidx + 1)%INPUT_SIZE;
       tidy ++;
    }

}


//--------------------------Class Functions-------------------------------------
__host__
__device__
void Neuron::update_input_weights(Layer *prev_layer)
{
    // Device-side wrapper that calls update_input_weights, parallelizing them.
    neuron_global_update_input_weights<<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (this, prev_layer);
    cudaDeviceSynchronize(); // Makes sure all threads finish before moving on
}
__host__
__device__
double Neuron::sum_DOW(Layer *next_layer)
{
    // Device-side wrapper that calculates the derivative of weights based on the error
    *DOW_sum = 0.0;
    // Sum our contributions of the errors at the nodes we feed.

    neuron_global_sum_DOW<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(this, DOW_sum, next_layer);
    cudaDeviceSynchronize(); // Waits for all threads to finish before moving on.
    return *DOW_sum;
}
__host__
__device__
void Neuron::calculate_hidden_gradients(Layer *next_layer)
{
    // Uses the derivative of weights to calculate the gradient, which is used to update the weights.
    double dow = sum_DOW(next_layer);
    gradient = dow * Neuron::transfer_function_derivative(output);
}

__host__
__device__
void Neuron::calculate_output_gradient(double target_val)
{
    // Calculates error, then uses it to determine output gradients.
    double delta = target_val - output;
    gradient = delta * Neuron::transfer_function_derivative(output);
}

__host__
__device__
double Neuron::transfer_function_derivative(double x)
{
    // Derivative transfer function to calculate derivative of weights
    return 1.0 - x * x;
}
__host__
__device__
double Neuron::transfer_function(double x)
{
    // Transfer function to determine the output value
    return tanh(x);
}
__host__
__device__
void Neuron::feed_forward(Layer *prev_layer)
{
    // Device-side wrapper for feed_forward.
    *FF_sum = 0.0;

    neuron_global_feed_forward<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(this, FF_sum, prev_layer);
    cudaDeviceSynchronize(); // Waits for all threads to synchronize.
    output = Neuron::transfer_function(*FF_sum);
}

__host__
__device__
Neuron::Neuron()
{
   my_index = 999;
}
__host__
__device__
 Neuron::Neuron(int num_connections, int index)
{
    // Neuron constructor. Somewhere in here, this breaks and exits silently, cancelling the
    //  rest of the run.
    output_weights = new Connection[num_connections];
    for (unsigned c = 0; c < num_connections; ++c){
        Connection connection;
        connection = Connection();

        connection.weight = 0.5;
        // Connection *cuda_connect;
        //
        // const size_t sz = sizeof(Connection);
        // cudaMalloc((void**)&cuda_connect, sz);
        // cudaMemcpy(cuda_connect, &connection, sz, cudaMemcpyHostToDevice);


        output_weights[c] = connection;
    }
    // These next lines break the program. We could not determine why, but assume it has to do
    //  with memory placement of these pointers.
    *DOW_sum = 0.0;
    *FF_sum = 0.0;
    my_index = index;
}


__host__
__device__
Layer::Layer()
{
    length = 0;
}
__host__
__device__
Layer::Layer(int num_neurons, int num_connections)
{
    // Creates the layer with however many neurons. Acts as an array.
    layer = new Neuron[num_neurons];
    for(int i=0;i<=num_neurons;i++){
        Neuron neuron;
        neuron = Neuron(num_connections, i);
        if(i == num_neurons-1){
            neuron.set_output(1.0);
        }

        // These lines were removed when we moved to creating objects on the GPU.
        //  All cudaMalloc and cudaMemcpy calls are strictly host functions.

        // Neuron *cuda_neuron;
        //
        // size_t sz = sizeof(Neuron);
        // cudaMalloc((void**)&cuda_neuron, sz);
        //
        // cudaMemcpy(cuda_neuron, &neuron, sz, cudaMemcpyHostToDevice);
        layer[i] = neuron;
    }
}


__host__
__device__
void Network::get_results(double *result_vals, int result_length)
{
    // Assigns the results of running the row to result_vals
    for(unsigned n = 0; n < result_length; ++n){
        Layer *output_layer = layers[NUM_HIDDEN_LAYERS+1];
        result_vals[n] = (output_layer->layer[n]->get_output());
        result_vals[n] = get_RAE();
    }
}

__host__
__device__
void Network::back_prop(double *target_vals, int target_length)
{
    // Network wrapper for the backpropagation function.
    Layer output_layer = layers[NUM_HIDDEN_LAYERS+1];
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

        net_global_backprop<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(&layers[layer_num], &layers[layer_num+1]);
        cudaDeviceSynchronize();
    }

    //For all layers from outputs to first hidden layer, update connection weights
    for(unsigned layer_num = NUM_HIDDEN_LAYERS+1;layer_num > 0; --layer_num){

        net_global_update_weights<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(&layers[layer_num], &layers[layer_num-1]);
        cudaDeviceSynchronize();
    }

}

__host__
__device__
void Network::feed_forward(double *input_vals, int input_vals_length)
{
    //assign the input values to the input neurons
    for(unsigned i = 0; i < input_vals_length; ++i){
        Layer input_layer = layers[0];
        input_layer.layer[i].set_output(input_vals[i]);
    }
    //forward prop
    for(unsigned num_layer = 1; num_layer < NUM_HIDDEN_LAYERS+2; ++num_layer){

        net_global_feed_forward<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(&layers[num_layer], &layers[num_layer-1]);
        cudaDeviceSynchronize();
    }
}


__host__
__device__
Network::Network()
{
    // Network initializer. All commended-out code contained host functions that are not useable in
    //  device functions, where this is being called.

    // cudaMallocManaged(&layers, sizeof(Layer *)*(NUM_HIDDEN_LAYERS+2));
    layers = new Layer[NUM_HIDDEN_LAYERS+2];

    Layer layer;

    layer = Layer(INPUT_SIZE, HIDDEN_SIZE); // This line never completed, as creating Neurons broke it

    // Layer *cuda_layer;
    //
    // size_t sz = sizeof(Layer);
    // cudaMalloc((void**)&cuda_layer, sz);
    // cudaMemcpy(cuda_layer, &layer, sz, cudaMemcpyHostToDevice);


    layers[0] = layer;
    // for (int i = 1; i<NUM_HIDDEN_LAYERS; i++) {
    //     layer = Layer(HIDDEN_SIZE, HIDDEN_SIZE);
    //     Layer *cuda_layer;
    //
    //     sz = sizeof(Layer);
    //     cudaMalloc((void**)&cuda_layer, sz);
    //     cudaMemcpy(cuda_layer, &layer, sz, cudaMemcpyHostToDevice);
    //     layers[i] = cuda_layer;
    // }
    layer = Layer(HIDDEN_SIZE, OUTPUT_SIZE);
    // Layer *cuda_layer_1;
    //
    // sz = sizeof(Layer);
    // cudaMalloc((void**)&cuda_layer_1, sz);
    // cudaMemcpy(cuda_layer_1, &layer, sz, cudaMemcpyHostToDevice);
    layers[NUM_HIDDEN_LAYERS] = layer;

    layer = Layer(OUTPUT_SIZE, 0);
    // Layer *cuda_layer_2;

    // sz = sizeof(Layer);
    // cudaMalloc((void**)&cuda_layer_2, sz);
    // cudaMemcpy(cuda_layer_2, &layer, sz, cudaMemcpyHostToDevice);
    layers[1 + NUM_HIDDEN_LAYERS] = layer;
    RAS = 100.0; //Number of training samples to average over


}




int main(){
    // Read the training data
    TrainingData trainData("faster_training_data.txt");
    cout << "CUDA BNN starting" << endl;

    // Create the space for the network...
    Network *cuda_net;
    const size_t sz = sizeof(Network);

    // ...and allocate that space in GPU memory.
    cudaMalloc((void**)&cuda_net, sz);

    double temp_inputs[INPUT_SIZE];
    double temp_targets[INPUT_SIZE];

    // Create the space for all of the inputs, expected outputs, and the place to store results
    double** input_array = (double **) malloc(TRAINING_SIZE * sizeof(double *));
    for(int i=0;i<TRAINING_SIZE;++i){
        input_array[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
    }
    double** target_array = (double **) malloc(TRAINING_SIZE * sizeof(double *));
    for(int i=0;i<TRAINING_SIZE;++i){
        target_array[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
    }
    double** result_array = (double **) malloc(TRAINING_SIZE * sizeof(double *));
    for(int i=0;i<TRAINING_SIZE;++i){
        result_array[i] = (double *)malloc(OUTPUT_SIZE * sizeof(double));
    }

    int index = 0;
    int training_pass = 0;

    cout << "Reading Training Data" << endl;

    while (!trainData.isEof()) {
        ++training_pass;

        // Get new input data and assign it to host memory
        trainData.getNextInputs(temp_inputs);
        for(int i=0;i<INPUT_SIZE;i++){
            input_array[index][i] = temp_inputs[i];
        }

        // Get the target data and assign it to host memory
        trainData.getTargetOutputs(temp_targets);
        for(int i=0;i<INPUT_SIZE;i++){
            target_array[index][i] = temp_targets[i];
        }
        index++;
    }

    // Allocate GPU memory and copy the host memory over. Pitch is required to properly align the data by padding memory, making for faster access
    //  on the GPU.

    size_t input_pitch;
    size_t target_pitch;
    size_t errors_pitch;
    double *inputs;
    double *targets;
    double *errors;
    cudaMallocPitch(&inputs, &input_pitch, sizeof(double)*INPUT_SIZE, TRAINING_SIZE);
    cudaMallocPitch(&targets, &target_pitch, sizeof(double)*INPUT_SIZE, TRAINING_SIZE);
    cudaMallocPitch(&errors, &errors_pitch, sizeof(double)*INPUT_SIZE, TRAINING_SIZE);
    cudaMemcpy2D(inputs, input_pitch, input_array, sizeof(double)*INPUT_SIZE, sizeof(double)*INPUT_SIZE, TRAINING_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy2D(targets, target_pitch, target_array, sizeof(double)*INPUT_SIZE, sizeof(double)*INPUT_SIZE, TRAINING_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy2D(errors, errors_pitch, result_array, sizeof(double)*INPUT_SIZE, sizeof(double)*INPUT_SIZE, TRAINING_SIZE, cudaMemcpyHostToDevice);

    dim3 gridSize(1,1); // Given other numbers than 1, would let us traverse 2d arrays with multiple threads. As is, results in a single thread.
    dim3 blockSize(1,1);

    cout << "Starting Training" << endl;

    // Run the training
    global_training<<<gridSize, blockSize>>>(cuda_net, inputs, targets, errors, input_pitch, target_pitch, errors_pitch);
    cudaDeviceSynchronize(); // make sure all threads finish before moving on


    cout << "I done running" << endl;
    cudaMemcpy2D(result_array, errors_pitch, errors, INPUT_SIZE*sizeof(double), INPUT_SIZE*sizeof(double), TRAINING_SIZE, cudaMemcpyDeviceToHost); // copy error back into host memory

    // Print results
    for(int i=0;i<TRAINING_SIZE;++i){
        cout << "Error: " ;
        cout << result_array[i][0] << " " ;

        cout << endl;
    }

    // Free all memory used.
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
