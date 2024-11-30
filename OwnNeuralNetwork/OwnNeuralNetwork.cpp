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

    size_t epochs = 800;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t j = 0; j < trainDataTable.getFilteredData().size(); ++j) {
            vector_type inputs = Helpers::ConvFunc(trainDataTable.getFilteredData()[j]);
            vector_type targets = Helpers::getEncoding(trainDataTable.getTargets()[j]);
            // scaling needed!
            nn.train(inputs, targets);
        }
    }

    auto corr_predictions = 0.0;
    size_t total_prdictions = 0;
    for (size_t j = 0; j < testDataTable.getFilteredData().size(); ++j) {
        vector_type inputs = Helpers::ConvFunc(testDataTable.getFilteredData()[j]);
        vector_type targets = Helpers::getEncoding(testDataTable.getTargets()[j]);

        // scaling needed!
        vector_type y = nn.query(inputs);

        // round to next int
        Eigen::VectorXd rounded_y = y.unaryExpr([](double v) { return std::round(v); });
        Eigen::VectorXd rounded_t = targets.unaryExpr([](double v) { return std::round(v); });

        auto dotProduct = rounded_t.dot(rounded_y);
        corr_predictions += dotProduct;

        std::cout << "y: " << std::endl;
        std::cout << rounded_y << std::endl;
        std::cout << "targets: " << std::endl;
        std::cout << rounded_t << std::endl;
        std::cout << "dotProduct: ";
        std::cout << dotProduct << std::endl;
        std::cout << std::endl;
        ++total_prdictions;
    }

    std::cout << "Correct predictions: " << corr_predictions << " of " << total_prdictions << std::endl;
    std::cout << "Score: " << corr_predictions/ total_prdictions << std::endl;

    // Endzeitpunkt erfassen
    auto end = std::chrono::high_resolution_clock::now();

    // Differenz berechnen
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Ergebnis ausgeben
    std::cout << "Die Berechnung hat " << duration.count() << " Millisekunden gedauert.\n";

    return 0;
}