#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono> // für Zeitmessung
#include <Eigen/Dense>

#include "getcsvcontent.h"
#include "metadata.h"
#include "data_table.h"
// #include "grouped_data.h"

#define _DOUBLE_

#ifndef _DOUBLE_
using decimal = float;
using vector_type = Eigen::VectorXf;
using matrix_type = Eigen::MatrixXf;
#else
using decimal = double;
using vector_type = Eigen::VectorXd;
using matrix_type = Eigen::MatrixXd;
#endif


//const std::string MetaDataFileName = "C:\\Users\\haasr\\source\\repos\\DataTreatmentStudy\\DataTreatmentStudy\\data\\irisMetaData.txt";
//const std::string CsvDataFileName = "C:\\Users\\haasr\\source\\repos\\DataTreatmentStudy\\DataTreatmentStudy\\data\\iris.csv";

const std::string MetaDataFileName = ".\\data\\irisMetaData.txt";
const std::string CsvDataFileName = ".\\data\\iris.csv";

namespace Helpers {
    template <typename T>
    T sigmoidFunction(T x) {
        return T();
    }

    // Spezialisierung für double
    template<>
    decimal sigmoidFunction(decimal x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    template<>
    vector_type sigmoidFunction(vector_type v) {
        vector_type res = v;
        for (size_t j = 0; j < res.size(); ++j) {
            res(j) = sigmoidFunction<decimal>(v(j));
        }
        return res;
    }

    vector_type ConvFunc(const std::vector<std::string>& _in) {
        size_t siz = _in.size();
        vector_type res(siz);
        for (size_t j = 0; j < siz; ++j) {
            try {
                res(j) = std::stod(_in[j]);
            }
            catch (std::exception ex) {
                // skip column, if its datatype cannot converted to a numeric format
            }
        }
        return res;
    }

    vector_type getEncoding(const std::string& _in) {
        decimal almostZero = 0.01;
        vector_type res{ {almostZero, almostZero, almostZero} };
        if (_in.find("Setosa") != std::string::npos) {
            res(0) = 1.0 - almostZero;
        }
        if (_in.find("Versicolor") != std::string::npos) {
            res(1) = 1.0 - almostZero;
        }
        if (_in.find("Virginica") != std::string::npos) {
            res(2) = 1.0 - almostZero;
        }
        return res;
    }

    size_t getCorrectPredictions(const std::vector<vector_type>& targets, const std::vector<vector_type>& predicted_targets) {
        // round to next int

        if (targets.size() != predicted_targets.size())
        {
            return SIZE_MAX;
        }

        size_t corr_predictions = 0;
        for (auto it = targets.begin(), it1=predicted_targets.begin(); it != targets.end(); ++it, ++it1) {
            Eigen::VectorXd rounded_it = it->unaryExpr([](double v) { return std::round(v); });
            Eigen::VectorXd rounded_it1 = it1->unaryExpr([](double v) { return std::round(v); });

            auto dotProduct = rounded_it.dot(rounded_it1);
            corr_predictions += dotProduct;
        }
        return corr_predictions;
    }
}

class NeuralNetwork {
public:
    NeuralNetwork(size_t _inputNodes, size_t _hiddenNodes, size_t _outputNodes, decimal _learningRate, std::function<vector_type(vector_type)> _activationFunction = Helpers::sigmoidFunction<vector_type>) :
        inputNodes(_inputNodes),
        hiddenNodes(_hiddenNodes),
        outputNodes(_outputNodes),
        learningRate(_learningRate),
        activationFunction(_activationFunction)
    {
        // build wInputHidden and wHiddenOutput as random matrices with normally distributed entries
        std::random_device rd{};
        std::mt19937 gen{ rd() };

        std::normal_distribution<decimal> distWInputHidden(0.0f, std::pow(inputNodes, -0.5f));
        std::normal_distribution<decimal> distWHiddenOutput(0.0f, std::pow(hiddenNodes, -0.5f));

        wInputHidden = matrix_type::NullaryExpr(hiddenNodes, inputNodes, [&]() {return distWInputHidden(gen); });
        wHiddenOutput = matrix_type::NullaryExpr(outputNodes, hiddenNodes, [&]() {return distWHiddenOutput(gen); });
    }

    vector_type query(vector_type _inputs) {
        // calculate signals into hidden layer
        auto hiddenInputs = wInputHidden * _inputs;
        // calculate the signals emerging from hidden layer
        auto hiddenOutputs = activationFunction(hiddenInputs);

        // calculate signals into final output layer
        auto finalInputs = wHiddenOutput * hiddenOutputs;
        // calculate the signals emerging from final output layer
        auto finalOutputs = activationFunction(finalInputs);

        return finalOutputs;
    }

    void train(vector_type _inputs, vector_type _targets) {

        // calculate signals into hidden layer
        auto hiddenInputs = wInputHidden * _inputs;
        // calculate the signals emerging from hidden layer
        auto hiddenOutputs = activationFunction(hiddenInputs);

        // calculate signals into final output layer
        auto finalInputs = wHiddenOutput * hiddenOutputs;
        // calculate the signals emerging from final output layer
        auto finalOutputs = activationFunction(finalInputs);

        // output layer error is the(target - actual)
        auto outputErrors = _targets - finalOutputs;
        // hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        auto hiddenErrors = wHiddenOutput.transpose() * outputErrors;

        // update the weights for the links between the hidden and output layers
        wHiddenOutput += learningRate * outputErrors.cwiseProduct(finalOutputs.cwiseProduct(vector_type::Constant(outputNodes, 1.0) - finalOutputs)) * hiddenOutputs.transpose();

        // update the weights for the links between the input and hidden layers
        wInputHidden += learningRate * hiddenErrors.cwiseProduct(hiddenOutputs.cwiseProduct(vector_type::Constant(hiddenNodes, 1.0) - hiddenOutputs)) * _inputs.transpose();
    }

    matrix_type getWInputHidden() const {
        return wInputHidden;
    }

    matrix_type getWHiddenOutput() const {
        return wHiddenOutput;
    }

// Methode für die accuracy
private:
    size_t inputNodes = 0;
    size_t hiddenNodes = 0;
    size_t outputNodes = 0;
    decimal learningRate = 0.0;
    matrix_type wInputHidden;
    matrix_type wHiddenOutput;
    std::function<vector_type(vector_type)> activationFunction;
};


int main()
{
    // Start der Zeitmessung
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::string>> content = getCsvContent(CsvDataFileName);

    DataTableMetaData dataTableMetaData;
    dataTableMetaData.setMetaData(MetaDataFileName);
    const size_t targetColumn = dataTableMetaData.getTargetColumn();
    const size_t firstLineToRead = dataTableMetaData.getFirstLineToRead();

    DataTable dataTable;
    dataTable.setMetaData(dataTableMetaData);
    dataTable.setData(content);
    dataTable.testTrainSplit(30);
    DataTable trainDataTable = dataTable.getTrainDataTable();
    DataTable testDataTable = dataTable.getTestDataTable();

    auto nn = NeuralNetwork(4, 4, 3, 0.12);
    //std::cout << "nn.wih =\n" << nn.getWInputHidden() << std::endl;
    //std::cout << "nn.who =\n" << nn.getWHiddenOutput() << std::endl;

    size_t test_data_size = testDataTable.getFilteredData().size();
    size_t epochs = 1600;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t j = 0; j < trainDataTable.getFilteredData().size(); ++j) {
            vector_type train_inputs = Helpers::ConvFunc(trainDataTable.getFilteredData()[j]);
            vector_type train_targets = Helpers::getEncoding(trainDataTable.getTargets()[j]);
            // scaling needed!
            nn.train(train_inputs, train_targets);

        }

        std::vector<vector_type> vector_predicted_test_targets(test_data_size);
        std::vector<vector_type> vector_test_targets(test_data_size);

        vector_predicted_test_targets.resize(test_data_size);
        vector_test_targets.resize(test_data_size);

        const size_t buf_size = 2;
        decimal accuracies[buf_size];

        for (size_t j = 0; j < test_data_size; ++j) {
            vector_type test_inputs = Helpers::ConvFunc(testDataTable.getFilteredData()[j]);
            vector_type test_targets = Helpers::getEncoding(testDataTable.getTargets()[j]);

            vector_type predicted_test_targets = nn.query(test_inputs);

            // round to next int
            predicted_test_targets = predicted_test_targets.unaryExpr([](double v) { return std::round(v); });
            test_targets = test_targets.unaryExpr([](double v) { return std::round(v); });

            vector_predicted_test_targets[j] = predicted_test_targets;
            vector_test_targets[j] = test_targets;

        }

        auto corr_predictions = Helpers::getCorrectPredictions(vector_test_targets, vector_predicted_test_targets);
        auto accuracy = static_cast<decimal>(corr_predictions) / static_cast<decimal>(test_data_size);
        accuracies[epoch % buf_size] = accuracy;
        
        std::cout << "Epoch: " << epoch << std::endl;
        std::cout << (epoch - 1) % buf_size << std::endl;
        std::cout << "Correct Predictions: " << corr_predictions << " out of " << test_data_size << std::endl;
        std::cout << "Accuracy: " << accuracy << std::endl;
        std::cout << std::endl;
        if (epoch < epochs/2) {
            continue;
        }
        if (accuracies[epoch % buf_size] < accuracies[(epoch - 1) % buf_size]) {
            break;
        }
    }

    // Endzeitpunkt erfassen
    auto end = std::chrono::high_resolution_clock::now();

    // Differenz berechnen
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Ergebnis ausgeben
    std::cout << "Die Berechnung hat " << duration.count() << " Millisekunden gedauert.\n";

    return 0;
}