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


class Layer : public Managed {
public:
    vector<Neuron> neurons;
};

// This is analogous to feedforward
__global__
void change_cat(Neuron &n_original, Neuron &n_to_copy) {
  n_original.my_cats = n_to_copy.my_cats;
}

void print_cats(Layer &layer) {
    for (int i = 0; i<layer.neurons.size(); i++) {
        printf("Neuron %i has %i\n", i+1, layer.neurons[i].my_cats);
    }
    printf("\n");
}

int main(void) {
    Neuron *n1 = new Neuron;    //New returns a pointer to the new object
    (*n1).my_cats = 2;
    Neuron *n2 = new Neuron;
    (*n2).my_cats = 7;
    Layer *layer = new Layer;
    (*layer).neurons.push_back(*n1);    //Should we store a vector of pointers or a vector of neurons?
    (*layer).neurons.push_back(*n2);
    
    print_cats(*layer);
    change_cat <<<1, 2>>> ((*n1), (*n2));
    cudaDeviceSynchronize();        //If you ever get a Bus Error, you probably forgot this line
    print_cats(*layer);
    (*n1).eh(3);
    print_cats(*layer);

    delete n1; delete n2; delete layer;
    return 0;
}