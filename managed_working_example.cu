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
    unsigned my_cat;
};


__global__
void change_cat(Neuron &n, unsigned new_val) {
  n.my_cat = new_val;
}

int main(void) {
    Neuron *n = new Neuron;
    (*n).my_cat = 2;
    
    printf("%i\n", (*n).my_cat);
    change_cat <<<1, 1>>> ((*n), 5);
    cudaDeviceSynchronize();
    printf("%i\n", (*n).my_cat);

    delete n;
    return 0;
}