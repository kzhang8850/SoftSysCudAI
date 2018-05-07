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
    TrainingData(const string filename);
    ~TrainingData(void);
    bool isEof(void) { return m_trainingDataFile.eof(); }

    // Returns the number of input values read from the file:
    void getNextInputs(double *inputVals);
    void getTargetOutputs(double *targetOutputVals);

private:
    ifstream m_trainingDataFile;
};
