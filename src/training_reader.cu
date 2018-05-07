#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include "training_reader.cuh"

using namespace std;
//-----------------------Training Class to load training data-------------------

// class TrainingData
// {
// public:
//     __host__ TrainingData(const string filename);
//     __host__ ~TrainingData(void);
//     __host__ bool isEof(void) { return m_trainingDataFile.eof(); }
//
//     // Returns the number of input values read from the file:
//     __host__ void getNextInputs(double *inputVals);
//     __host__ void getTargetOutputs(double *targetOutputVals);
//
// private:
//     ifstream m_trainingDataFile;
// };

__host__
TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

__host__
TrainingData::~TrainingData()
{
    m_trainingDataFile.close();
}

__host__
void TrainingData::getNextInputs(double *inputVals)
{
    int index = 0;
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals[index] = oneValue;
            ++index;
        }
    }
}

__host__
void TrainingData::getTargetOutputs(double *targetOutputVals)
{
    int index = 0;
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals[index] = oneValue;
            ++index;
        }
    }

}
