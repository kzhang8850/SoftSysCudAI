#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
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

class Neuron : public Managed {
public:
    // This is analogous to output_val
    unsigned my_cats;
    // This is analogous to setOutputVals
    void eh(unsigned new_num) {my_cats = new_num;};
};

// This is analogous to feedforward
__global__
void change_cat(Neuron &n_original, Neuron &n_to_copy) {
  n_original.my_cats = n_to_copy.my_cats;
}

int main(void) {
    Neuron *n1 = new Neuron;
    (*n1).my_cats = 2;
    Neuron *n2 = new Neuron;
    (*n2).my_cats = 7;

    printf("Neuron1: %i and Neuron2: %i\n", (*n1).my_cats, (*n2).my_cats);
    change_cat <<<1, 1>>> ((*n1), (*n2));
    cudaDeviceSynchronize();
    printf("Neuron1: %i and Neuron2: %i\n", (*n1).my_cats, (*n2).my_cats);
    (*n1).eh(3);
    printf("Neuron1: %i and Neuron2: %i\n", (*n1).my_cats, (*n2).my_cats);

    delete n1; delete n2;
    return 0;
}