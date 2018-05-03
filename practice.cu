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

using namespace std;

int main(){
	int *thing;
	size_t pitch;
	
	for (int i=0; i<6; i++){
		for (int j=0; j<2; j++){
			thing[i][j] = i + j;
		}
	}
	for (int i=0; i<6; i++){
		for (int j=0; j<2; j++){
			printf("%i ", thing[i][j]);
		}
		printf("\n");
	}
	printf("%i\n", *(*(thing + 1) + 1));
}