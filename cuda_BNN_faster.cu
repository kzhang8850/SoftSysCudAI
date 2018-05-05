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


int iDivUp(int hostPtr, int b){
    return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }


__global__ void test_kernel_2D(double *devPtr, size_t pitch, double *target, size_t target_pitch)
{
   int tidx = blockIdx.x*blockDim.x + threadIdx.x;
   int tidy = blockIdx.y*blockDim.y + threadIdx.y;
   while ((tidy < TRAINING_SIZE))
   {
       double *row_a = (double *)((char*)devPtr + tidy * pitch);
       printf("Inputs: %f\n", row_a[tidx]);
       double *row_b = (double *)((char*)target + tidy * target_pitch);
       if(tidx == 0){
          printf("Outputs: %f\n", row_b[tidx]);

       }

       tidx = (tidx + 1)%INPUT_SIZE;
       tidy ++;
    }

}


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

class Connection
{
public:
    double weight;
    double delta_weight;
};


class Neuron
{
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
    // __host__ __device__ double init_weight(double *result);
    __host__ __device__ double sum_DOW(Layer *next_layer);


};



class Layer
{
public:
    __host__ __device__ Layer();
    __host__ __device__ ~Layer(){};
    __host__ __device__ Layer(int num_neurons, int num_connections);
    Neuron* layer;
    int length;
};


class Network
{
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

        prev_neuron.output_weights[neuron->my_index].delta_weight = new_delta_weight;
        // cout << "DELTA WEIGHT " << new_delta_weight << endl;
        prev_neuron.output_weights[neuron->my_index].weight += new_delta_weight;

        // cout << "WEIGHT: " << prev_neuron.output_weights[neuron.my_index].weight << endl;
    }

}

__global__
void net_global_feed_forward(Layer *l, Layer *pl)
{
    Layer layer = *l;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index; i < layer.length-1;i +=stride){
        layer.layer[i].feed_forward(pl);
    }

}

__global__
void net_global_update_weights(Layer *l, Layer *pl)
{
    Layer layer = *l;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i=index; i < layer.length-1;i+=stride){
        layer.layer[i].update_input_weights(pl);
    }

}

__global__
void net_global_backprop(Layer *hl, Layer *nl)
{
    Layer hidden_layer = *hl;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n = index; n < hidden_layer.length; n+=stride) {
        hidden_layer.layer[n].calculate_hidden_gradients(nl);
    }

}

__global__
void global_training(Network *net, double* inputs, double* targets, double* errors, size_t input_pitch, size_t target_pitch, size_t errors_pitch)
{
    Network network = *net;
    network = Network();
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    while (tidy < TRAINING_SIZE)
    {
       double *row_a = (double *)((char*)inputs + tidy * input_pitch);
       network.feed_forward(row_a, INPUT_SIZE);
    //    printf("Inputs: %f\n", row_a[tidx]);
       double *row_b = (double *)((char*)errors + tidy * errors_pitch);
       double *row_c = (double *)((char*)targets + tidy * target_pitch);

       network.get_results(row_b, OUTPUT_SIZE);
       network.back_prop(row_c, OUTPUT_SIZE);

       printf("Error: %f", network.get_RAE());


       // if(tidx == 0){
       //    printf("Outputs: %f\n", row_b[tidx]);

       // }

       tidx = (tidx + 1)%INPUT_SIZE;
       tidy ++;
    }
    printf("I'm done");

}


//--------------------------Class Functions-------------------------------------
__host__
__device__
void Neuron::update_input_weights(Layer *prev_layer)
{

    neuron_global_update_input_weights<<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (this, prev_layer);
    cudaDeviceSynchronize();
}
__host__
__device__
double Neuron::sum_DOW(Layer *next_layer)
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
void Neuron::calculate_hidden_gradients(Layer *next_layer)
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
void Neuron::feed_forward(Layer *prev_layer)
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
__device__
Neuron::Neuron()
{
   my_index = 999;
}
__host__
__device__
 Neuron::Neuron(int num_connections, int index)
{
    // cudaMallocManaged(&output_weights, sizeof(Connection *)*num_connections);
    output_weights = new Connection[num_connections];
    for (unsigned c = 0; c < num_connections; ++c){
        Connection connection;
        printf("I'm in Connection\n");
        connection = Connection();
        printf("I'm done\n");

        connection.weight = 0.5;
        // Connection *cuda_connect;
        //
        // const size_t sz = sizeof(Connection);
        // cudaMalloc((void**)&cuda_connect, sz);
        // cudaMemcpy(cuda_connect, &connection, sz, cudaMemcpyHostToDevice);


        output_weights[c] = connection;
    }
    // cudaMallocManaged(&DOW_sum, sizeof(double));
    // cudaMallocManaged(&FF_sum, sizeof(double));
    printf("I'm hello");
    *DOW_sum = 0.0;
    *FF_sum = 0.0;
    printf("I'm hello\n");
    my_index = index;
    printf("I'm hello\n");
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
    // cudaMallocManaged(&layer, sizeof(Neuron *)*num_neurons);
    layer = new Neuron[num_neurons];
    for(int i=0;i<=num_neurons;i++){
        Neuron neuron;
        printf("I'm in neuron\n");
        neuron = Neuron(num_connections, i);
        printf("I'm done\n");
        if(i == num_neurons-1){
            neuron.set_output(1.0);
        }
        // Neuron *cuda_neuron;
        //
        // size_t sz = sizeof(Neuron);
        // cudaMalloc((void**)&cuda_neuron, sz);
        //
        // cudaMemcpy(cuda_neuron, &neuron, sz, cudaMemcpyHostToDevice);
        layer[i] = neuron;
    }

    // layer[num_neurons-1]->set_output(1.0);
    // Neuron *chicken = layer[num_neurons-1];
    // Neuron chicken2 = *chicken;
    // chicken->set_output(1.0);
    // cout << "hello" << endl;

    length = num_neurons+1;
    // exit(0);
}


__host__
__device__
void Network::get_results(double *result_vals, int result_length)
{
    for(unsigned n = 0; n < result_length; ++n){
        // Layer *output_layer = layers[NUM_HIDDEN_LAYERS+1];
        // printf("WUT THIS %f\n", get_RAE());
        // result_vals[n] = (output_layer->layer[n]->get_output());
        // result_vals[n] = get_RAE();
    }
}

__host__
__device__
void Network::back_prop(double *target_vals, int target_length)
{
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
    // cudaMallocManaged(&layers, sizeof(Layer *)*(NUM_HIDDEN_LAYERS+2));
    layers = new Layer[NUM_HIDDEN_LAYERS+2];

    Layer layer;
    printf("hello\n");

    layer = Layer(INPUT_SIZE, HIDDEN_SIZE);
    printf("I'm done");

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
    TrainingData trainData("faster_training_data.txt");
    cout << "CUDA BNN starting" << endl;

    // Network myNet = Network();

    Network *cuda_net;
    const size_t sz = sizeof(Network);

    cudaMalloc((void**)&cuda_net, sz);
    // cudaMemcpy(cuda_net, &myNet, sz, cudaMemcpyHostToDevice);
    double temp_inputs[INPUT_SIZE];
    double temp_targets[INPUT_SIZE];

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


    // cudaMalloc(&input_array, sizeof(double)*TRAINING_SIZE);
    // cudaMalloc(&target_array, sizeof(double)*TRAINING_SIZE);
    // cudaMalloc(&result_array, sizeof(double)*TRAINING_SIZE);
    int index = 0;
    int training_pass = 0;

    cout << "Reading Training Data" << endl;

    while (!trainData.isEof()) {
        ++training_pass;
        // cout <<  "Pass " << training_pass << endl;

        // Get new input data and feed it forward:
        trainData.getNextInputs(temp_inputs);
        for(int i=0;i<INPUT_SIZE;i++){
            input_array[index][i] = temp_inputs[i];
            // cout << temp_inputs[i] << ":  ";
        }

        // cout << endl;

        // cout << "Put inputs into input_array" << endl;

        // Get new input data and feed it forward:
        // showVectorVals(": Inputs:", input_vals, INPUT_SIZE);
        // myNet.feed_forward(input_vals, INPUT_SIZE);

        // Collect the net's actual output results:
        // myNet.get_results(result_vals, OUTPUT_SIZE);
        // showVectorVals("Outputs:", result_vals, OUTPUT_SIZE);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(temp_targets);
        for(int i=0;i<INPUT_SIZE;i++){
            target_array[index][i] = temp_targets[i];
        }
        // cout << "Put targets into target_array" << endl;
        index++;
        // cout<<index<<endl;
        // showVectorVals("Targets:", target_vals, OUTPUT_SIZE);
        // myNet.back_prop(target_vals, OUTPUT_SIZE);

        // Report how well the training is working, average over recent samples:
        // cout << "Net recent average error: " << myNet.get_RAE() << endl;
    }
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

    // dim3 gridSize(iDivUp(INPUT_SIZE, BLOCKSIZE_x), iDivUp(TRAINING_SIZE, BLOCKSIZE_y));
    // dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

    dim3 gridSize(1,1);
    dim3 blockSize(1,1);

    // test_kernel_2D << <gridSize, blockSize >> >(inputs, input_pitch, targets, target_pitch);
    // cudaDeviceSynchronize();

    cout << "Starting Training" << endl;
    global_training<<<gridSize, blockSize>>>(cuda_net, inputs, targets, errors, input_pitch, target_pitch, errors_pitch);
    cudaDeviceSynchronize();


    cout << "I done running" << endl;
    cudaMemcpy2D(result_array, errors_pitch, errors, INPUT_SIZE*sizeof(double), INPUT_SIZE*sizeof(double), TRAINING_SIZE, cudaMemcpyDeviceToHost);

    // for(int i=0;i<TRAINING_SIZE;++i){
    //     cout << "Error: " ;
    //     cout << result_array[i][0] << " " ;
    //
    //     cout << endl;
    //
    //
    // }


    // cudaFree(inputs);
    // cudaFree(targets);
    // cudaFree(errors);
    //
    // for(int i=0;i<TRAINING_SIZE;++i){
    //     free(input_array[i]);
    // }
    // for(int i=0;i<TRAINING_SIZE;++i){
    //     free(target_array[i]);
    // }
    // for(int i=0;i<TRAINING_SIZE;++i){
    //     free(result_array[i]);
    // }
    // free(input_array);
    // free(target_array);
    // free(result_array);

    cout << endl << "Done!" << endl;
    return 0;
}
