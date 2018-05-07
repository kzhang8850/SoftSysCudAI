#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

//-----------------------Training Class to load training data-------------------

class TrainingData
{
public:
    __host__ TrainingData(const string filename);
    __host__ ~TrainingData(void);
    __host__ bool isEof(void) { return m_trainingDataFile.eof(); }

    // Returns the number of input values read from the file:
    __host__ void getNextInputs(double *inputVals);
    __host__ void getTargetOutputs(double *targetOutputVals);

private:
    ifstream m_trainingDataFile;
};
