#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include "training_reader.h"

using namespace std;

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

TrainingData::~TrainingData()
{
    m_trainingDataFile.close();
}

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
