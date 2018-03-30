## CudAI
#### Sean Foey, Kaitlyn Keil, Kevin Zhang

### Project Goal

For our project, we intend to implement machine learning using CUDA programming. We plan to make a machine learning algorithm that can accomplish a basic task and power it with our graphics card. It should be able to predict potentially a classification task and handle a large quantity of data for training fairly quickly. A concrete example might be crunching through hundreds of thousands of description data for various houses and predicting whether that house exists in San Francisco or New York. A lower bound for our project might be to implement a very basic machine learning algorithm, potentially just using some sort of arithmetic task, and making it able to train through lots of data quickly and correctly predict solutions. Stretch goals would include increasing the amount of data, and potentially evolving the task to become more complicated (i.e. more features and descriptions). We could also make our project approachable and informative enough that other people use our repo to learn about CUDA and ML.

### Learning Goals

We will learn more about CUDA programming and machine learning, figure out how to use our graphics card in boosting program performance, and learn some C++ along the way as a bonus.

### Resources

The first thing we will need to do, as reflected in the steps we describe below, are find installation instructions for CUDA and setting up graphics card stuff, which probably involves tutorials to get started with. We will initially try to find these on our own, as there are so many resources online, and reach out for help if we are still struggling. Kevin has made a start on getting things set up on his computer, and so can probably save us from some of the common pitfalls.

Nvidia has tons of documentation online on its website, including various tutorials on how to install CUDA, one linked below:
https://developer.nvidia.com/cuda-toolkit

There’s also another one by Tensorflow for GPU support, which uses Cuda
https://www.tensorflow.org/install/install_linux

Cuda programming itself also has tons of tutorials online from more experienced programmers, one included below:
https://devblogs.nvidia.com/even-easier-introduction-cuda/

### First Steps

Most of the first steps involve setting up our systems to be able to handle this project and feeling out a new space, as none of us have previously worked with CUDA programming. Before we can fully start in on our algorithm, then, we have some necessary steps:

- Install CUDA. This is primarily Sean and Kaitlyn, who have not yet attempted it. Everyone will need to test their environment. The definition of done for this will be testing Cuda on our computers with a basic Cuda program test script and confirming that we can run it.
- Find resources to get us started with CUDA programming and machine learning. All of us will need to find something that works well for us (tutorial or otherwise), and we can share our findings. The definition of done for this will be having read a couple article and websites on CUDA and machine learning, and going over a few scripts, potentially writing a few very basic ones by ourselves.
- Determine what sort of data we will be processing with our algorithm. Right now, our project is fairly nebulous, and we haven’t determined what exactly this algorithm is going to be based around. This could look like each of us coming to the next meeting with at least one idea and presenting them around before deciding, at which point we can update our proposal to reflect the plan. The definition of done for this will be putting down a solidified plan for what task and what data we want to solve with machine learning in CUDA.
