all: cuda_BNN_faster cuda_BNN

cuda_BNN: cuda_BNN.cu training_reader.cu training_reader.cuh
	nvcc -arch=sm_35 -rdc=true cuda_BNN.cu training_reader.cu -o cuda_BNN -lcudadevrt
cuda_BNN_faster: cuda_BNN_faster.cu training_reader.cuh training_reader.cu
	nvcc -arch=sm_35 -rdc=true cuda_BNN_faster.cu training_reader.cu -o cuda_BNN_faster -lcudadevrt
