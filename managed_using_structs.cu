#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
using namespace std;

struct Neuron {
  unsigned my_cat;
};

__global__
void change_cat(Neuron &n, unsigned new_val) {
  n.my_cat = new_val;
}

int main(void) {
    Neuron *n_ptr;
    cudaMallocManaged(&n_ptr, sizeof(Neuron));
    (*n_ptr).my_cat = 1;
    printf("%i\n", (*n_ptr).my_cat);
    change_cat <<<1, 1>>> ((*n_ptr), 2);
    cudaDeviceSynchronize();
    printf("%i\n", (*n_ptr).my_cat);
    return 0;
}