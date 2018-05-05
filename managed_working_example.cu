#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
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

class Neuron;

class Neuron : public Managed {
public:
    // This is analogous to output_val
    unsigned my_cats;
    // This is analogous to setOutputVals
    __device__ __host__ void eh(unsigned new_num); //You have to put tags here as well, or CUDA won't compile it
};

//This gets highlighted weirdly for me, but it works - you can call it from the CPU or GPU
__host__
__device__
void Neuron:: eh(unsigned new_num) {
    Neuron(int my_cats);
    my_cats = new_num;
};

class Layer : public Managed {
public:
    //Vector of pointers to Neuron objects
    vector<Neuron *> neurons;
};

// This is analogous to feedforward
// It just takes my_cats from one neuron and sticks it in another
__global__
void change_cat(Neuron &n_original, Neuron &n_to_copy) {
  n_original.my_cats = n_to_copy.my_cats;
  tanh(2.0);
}

//Playing with Eh as a device function; it has to be called from a global. That's what this wrapper is for.
__global__ 
void eh_wrapper(Neuron &n, unsigned new_num) {
    n.eh(new_num);
}

//Print out the my_cats value for every neuron in the layer
void print_cats(Layer &layer) {
    for (int i = 0; i<layer.neurons.size(); i++) {
        printf("Neuron %i has %i\n", i+1, (*(layer.neurons[i])).my_cats);
    }
    printf("\n");
}

int main(void) {
    Neuron *n1 = new Neuron(2);    //New returns a pointer to the new object
    // (*n1).my_cats = 2;
    Neuron *n2 = new Neuron(7);
    // (*n2).my_cats = 7;
    Layer *layer = new Layer;

    (*layer).neurons.push_back(n1);    //Should we store a vector of pointers or a vector of neurons?
    (*layer).neurons.push_back(n2);    //Answer: Vector of neurons didn't work, vector of pointers did
    print_cats(*layer);

    change_cat <<<1, 2>>> ((*n1), (*n2));
    cudaDeviceSynchronize();        //If you ever get a Bus Error, you probably forgot this line
    print_cats(*layer);

    eh_wrapper <<<1, 1>>> ((*n1), 3);
    cudaDeviceSynchronize();
    print_cats(*layer);

    delete n1; delete n2; delete layer;
    return 0;
}