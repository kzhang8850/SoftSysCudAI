## CudAI

#### Sean Foley, Kaitlyn Keil, Kevin Zhang

### The Goal
This project is an exploration into machine learning and CUDA programming. We intend to make a vanilla machine learning algorithm using only vectors that accomplish a basic task and then parallelize it with our graphics card’s computational power. Specifically, this takes the form of a backpropagation neural network (BNN) that can start from nothing and learn how to do a simple classification task, such as predicting XOR outputs. A possible stretch goal is increasing the complexity of our BNN to make predictions on data with more features, such as the MNIST digits dataset or house price prediction. We also write our code as approachable as we can, and informative enough that other people can use our repo to learn about CUDA and ML.

### Learning Objectives
We wanted to learn about the theory behind machine learning--specifically neural networks--as well as its physical implementation in code. We also wanted to understand the basics behind how parallelized computing works, which includes learning the popular graphics programming API CUDA and seeing how it boosts performance, if only on a small scale. In addition, we also wanted to expand on our current knowledge of C and jump into the more practical world of C++.

### Our Adventure, and our Results
Our project diverges into two paths. The first is a working non-parallel neural network in C++. It is structured by classes--specifically, a Network, a number of Layers, and several Neurons per layer, each of which shares a Connection struct with all the others in the neighboring layer (see Fig. 1 below). This network reports a  final accuracy of roughly *NUMBER HERE* on an XOR data set.

##### Figure 1: Code Architecture

![Code Architecture Animation](https://media.giphy.com/media/xlCHGS3xU5g0SwPbdf/giphy.gif)

Transfering from C++ to CUDA, we build a number of simple programs to help understand CUDA programming. For instance, [`managed_working_example`](https://github.com/kzhang8850/SoftSysCudAI/blob/master/managed_working_example.cu) is a testbed for us to understand the [unified memory system](https://devblogs.nvidia.com/unified-memory-in-cuda-6/).

Integration of these two things brings up an number of peculiarities of CUDA programming. From our research, it seems that there are not many others who have tried to use nested classes with CUDA. There seem to be two main reasons for this. First, for CUDA code to be fast, data and functions need to be loaded locally onto the GPU. We need to explicitly allocate space on the correct device for data, and tag functions as GPU or CPU specific. As a result, basic functions like C++ vector operations are not available on the GPU. We have to explicitly create most functions we want on the GPU. The second reason for avoiding nested classes builds off of that. Creating object oriented code becomes very complicated when you have to be explicit about the memory locations of functions. In particular, initializing custom objects containing other objects is tricky.

By inundating our class definitions with `cudaMallocManaged` calls and jumping back and forth between host and device, we are able to create a neural network that uses multiple threads on the GPU to train. Running it against a profiler, we get the results seen in Fig <INSERT RESULTS FIGURE NUMBER HERE>. The first thing that stands out is how much slower this is, despite parallelization. The time sink here is due to how often we try to port between device and host, which is a costly switch. Most kernels called in CUDA programs only copy over memory and enter once, to prevent this slowdown.

In light of this, we have `cuda_BNN_faster.cu`, which is meant to be an optimized version. We intend to initialize everything on the CPU, then pass values to the GPU and let it go from there, with no switching. However, this dive into structuring classes to be used inside `__global__` functions reveals a flaw in our original implementation of the nested classes. Because they all inherited from a class we call `Managed`, which sets aside managed memory for each instance, all of our classes are constructed using a `__host__` function, `cudaMallocManaged`. While this was not a problem when we set everything up on the host side, trying to initialize the network on the GPU could not work, instead exiting silently without completing the requested functionality. Because of this, and due to how we would have to restructure our program, the classes give us nothing that functions would not do better in this setup. 

While we set out hoping to unify C++ and CUDA, we realize that the two languages, despite sharing many similarities, want to operate in different ways. Specifically, while C++ does well in an object-oriented environment, writing functional CUDA code is a more effective approach than trying to shoehorn it into an object oriented architecture. CUDA shines best when there are a lot of computations that can take place all at once; while our implementation of BNN does have this happening on a layer-by-layer basis, we also have many places that things must happen sequentially, which removes some of the effectiveness of the GPU.


### Reflection and Final Thoughts
While our final product is not exactly what we hoped for, we did make a great deal of progress toward our learning goals. We especially learned a lot about the layout of GPU blocks and how to efficiently access GPU memory. We also were able to highlight the differences between C++ and CUDA.

To different extents, we achieved our goal of learning both C++ and CUDA as languages. While some of us got more hands-on experience due to functioning environments, we ensured that all of our team understood the code and why it was structured as it was.

### Resources and Useful Links
#### CUDA Basics
[An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/) This is a basic introduction to how CUDA works, include elements like tagging functions, memory allocation for the GPU (in this case, unified memory), synchronization, and how to call kernels. It also makes for a good test over whether all elements are working.
[Unified Memory in CUDA 6](https://devblogs.nvidia.com/unified-memory-in-cuda-6/) While the introduction to CUDA mentioned functions like `cudaMallocManaged` and unified memory, the details remained fuzzy for us. Particularly when we started defining classes that we wanted to be accessible on both the CPU and the GPU, we needed this more detailed explanation of how unified memory worked and ways to use it in inheritance.
 
#### Neural Networks
[15 Steps to Implement a Neural Net](http://code-spot.co.za/2009/10/08/15-steps-to-implemented-a-neural-net/) A fairly high-level walkthrough of creating a backpropogation neural network. This is valuable for wanting a more general picture of what the steps are, without having the specifics defined.
[A Neural Network in 10 Lines of C++ Code](https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/) and sister article [A Neural Network in 10 Lines of CUDA C++ Code](https://cognitivedemons.wordpress.com/2017/09/02/a-neural-network-in-10-lines-of-cuda-c-code/) These articles are liars, as the final file ends up being much more than 10 lines, but the core of the learning algorithm is short and simple. They provide a basic introduction to the math behind neural networks and a good insight into the differences between CUDA and straight C++.
[David Miller’s Neural Network in C++ Tutorial](https://vimeo.com/19569529) We use this tutorial, and the basic structure that it set for neurons, layers, and a network, as the backbone of our own BNN. Though rather long, it provides a good breakdown of all the different parts, and a more object-centered way of thinking about the network than most matrix-based tutorials.
