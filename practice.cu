#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <curand.h>
#include <curand_kernel.h>
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


class Blah : public Managed{
public:
  int thing;
};

class Blah_Container : public Managed{
public:
  Blah_Container(int num);
  Blah* blahs;
};

Blah_Container::Blah_Container(int num) {
  blahs = new Blah[num];
  for (int i=0; i<num; i++) {
    blahs[i].thing = i;
  }
}


// class Managed {
// public:
//     void *operator new(size_t len) {
//         void *ptr;
//         cudaMallocManaged(&ptr, len);
//         cudaDeviceSynchronize();
//         return ptr;
//     }

//     void operator delete(void *ptr) {
//         cudaDeviceSynchronize();
//         cudaFree(ptr);
//     }
// };

// // class Neuron : public Managed {
// // public:
// //     static unsigned my_cat;
// // private:
// //     unsigned m_myIndex;
// //     double m_gradient;
// // };

// struct Neuron {
//   unsigned my_cat;
// };

// __global__
// void change_cat(Neuron &n, unsigned new_val) {
//   n.my_cat = new_val;
// }

// __device__
// int scale_the_num(int num, int scale) {
//   return num*scale;
// }

// __global__
// void scale_nums(int* nums, int* scaled_nums, int scale) {
//   scaled_nums[threadIdx.x] = scale_the_num(nums[threadIdx.x], scale);
// }

// __global__
// void change_N(int *N)
// {

// }

__device__
void print_it(Blah &blah) {
  printf("%i\n", blah.thing);
}

__global__
void print_wrap(Blah_Container &cont){
  print_it(cont.blahs[threadIdx.x]);
}

int main(void) {
  // Blah_Container* um = new Blah_Container(5);
  // for (int i=0; i<5; i++) {
  //     printf("%i\n", um->blahs[i].thing);
  // }
  // print_wrap <<<1, 5>>> (*um);
  // cudaDeviceSynchronize();
  int sum = 0;
  int *sum1 = &sum;
  *sum1 = *sum1 + 1;
  cout << *sum1 << endl;

  // double thing[];
  // thing = {1, 2, 3};
  // thing.back()
  // Messing with number of threads
  // int *N;
  // cudaMallocManaged(&N, sizeof(int));
  // *N = 100;
  // printf("N starts as %i\n", *N);
  // change_N <<<1, 9>>> (N);
  // cudaDeviceSynchronize();
  // printf("Now it's %i\n", *N);
  // cudaFree(N);

  // int N = 20;
  // int scale = 2;

  // Neuron *n_ptr;
  // cudaMallocManaged(&n_ptr, sizeof(Neuron));
  // (*n_ptr).my_cat = 1;
  // printf("%i\n", (*n_ptr).my_cat);
  // change_cat <<<1, 1>>> ((*n_ptr), 2);
  // cudaDeviceSynchronize();
  // printf("%i\n", (*n_ptr).my_cat);
  // // int* nums;
  // // int* scaled_nums;
  // // cudaMallocManaged(&nums, sizeof(int)*N);
  // // cudaMallocManaged(&scaled_nums, sizeof(int)*N);

  // for (int i=0; i<N; i++)
  //   nums[i] = i;

  // scale_nums <<<1, N>>> (nums, scaled_nums, scale);
  // cudaDeviceSynchronize();

  // printf("[");
  // for (int i = 0; i < N; i++)
  //   printf("%i  ", scaled_nums[i]);
  // printf("]\n");

  // cudaFree(nums);
  // cudaFree(scaled_nums);

  return 0;
}
