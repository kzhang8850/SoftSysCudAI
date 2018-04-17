## CudAI

#### Sean Foley, Kaitlyn Keil, Kevin Zhang

### The Final Goal

This project is an exploration into machine learning and CUDA programming. We plan to make a vanilla machine learning algorithm using only vectors that can accomplish a basic task and parallelize it with our graphics card’s computational power. Concretely, this would take the form of a backpropagation neural network (BNN) that can perform basic classification tasks.  Our MVP will be a 3-4 layer BNN that can predict where a house’s location is based on various attributes such as square footage and number of bathrooms. A stretch goal might be to increase the complexity of our BNN to make predictions on data with more features, such as the MNIST digits dataset. We also plan to make our project approachable and informative enough that other people use our repo to learn about CUDA and ML.

### What We'll Learn

We want to learn about the theory behind machine learning as well as its physical implementation in code. We also want to understand the basics behind how parallelized computing works, which includes learning the popular graphics programming API CUDA and see how it boosts performance, if only on a small scale. In addition, we also wanted to expand on our current knowledge of C and jump into the more practical world of C++, therefore we want everything to be written solely in C++.

Whether or not this was an initial learning goal, this project helps us understand memory and how it is handled between the GPU and CPU, as well as the limitations of GPU functions. These limitations lead to learning more about class inheritance, overriding functions, and other fun aspects of OOP in C++.

### Steps Thus Far

Once everything was properly installed and running as expected, we started by finding a number of tutorials; some centered on [CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/), some on [neural networks](http://code-spot.co.za/2009/10/08/15-steps-to-implemented-a-neural-net/), and some bridged the gap (such as these two by Cognitive Demons, which walk through a simple neural net with [C++](https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/) and then with [CUDA](https://cognitivedemons.wordpress.com/2017/09/02/a-neural-network-in-10-lines-of-cuda-c-code/)). David Miller’s [video on C++ BNNs](https://vimeo.com/19569529) laid a base for our C++ foundation, which we are currently transfering into using the GPU. For this, we’ve also learned about [unified memory](https://devblogs.nvidia.com/unified-memory-in-cuda-6/). We think that, with all of these and with regular testing, we can make a fully-fledged neural network that parallelizes the computation of each layer of neurons.

### What Comes Next

We’re currently ironing out the details of parallelization. We reached our goals of a working, non-parallel BNN and several simple testbeds to experiment with CUDA programming. Going forward, we are applying what we’ve learned from the testbeds to our BNN. We have a handful of important immediate tasks to complete:

- Finish our CUDA managed memory implementation, and make sure all of our classes are properly allocated on both host and device memory.
- Correctly modify and tag our functions as host, global, and device functions, so CUDA knows how to run them.
- Document the working code we have so far.

Everyone is now working together on all the above tasks, and we trade off pair programming with some progress made individually on our own time. We’ll know the first two tasks are done when we have code that compiles and successfully learns the XOR task, which is our current simple task to get a parallelized BNN up and running, and the third is a constant work in progress. Finally, we have some tasks that are farther into the future:

- Speed up the program by implementing shared memory. We’ll quantify the change using CUDA’s built-in profiler, nvprof.
- Adapt the completed BNN framework to work with a more complex dataset, such as housing.

The above two tasks are also considered shared by everyone on the team in the same manner as before. The first will be considered done if a significant speedup is discovered in our runtimes with nvprof. The second will be considered done when we can successfully compile and run our BNN on the new dataset with accurate predictions.