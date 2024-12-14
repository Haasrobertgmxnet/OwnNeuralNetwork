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

#include "nn_defs.h"
#include "helpers.h"

const std::string MetaDataFileName = ".\\data\\irisMetaData.txt";
const std::string CsvDataFileName = ".\\data\\iris.csv";

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
    auto nn_ws = nn;
    //std::cout << "nn.wih =\n" << nn.getWInputHidden() << std::endl;
    //std::cout << "nn.who =\n" << nn.getWHiddenOutput() << std::endl;

    size_t test_data_size = testDataTable.getFilteredData().size();
    size_t epochs = 50;

    const uint8_t patience_const = 10;
    uint8_t patience = patience_const;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t j = 0; j < trainDataTable.getFilteredData().size(); ++j) {
            vector_type train_inputs = Helpers::ConvFunc(trainDataTable.getFilteredData()[j]);
            vector_type train_targets = Helpers::getEncoding(trainDataTable.getTargets()[j]);
            // scaling needed!
            nn_ws.train(train_inputs, train_targets);
        }

        std::vector<vector_type> vector_predicted_test_targets(test_data_size);
        std::vector<vector_type> vector_test_targets(test_data_size);      

        for (size_t j = 0; j < test_data_size; ++j) {
            vector_type test_inputs = Helpers::ConvFunc(testDataTable.getFilteredData()[j]);
            vector_type test_targets = Helpers::getEncoding(testDataTable.getTargets()[j]);

            vector_type predicted_test_targets = nn_ws.query(test_inputs);

            // round to next int
            predicted_test_targets = predicted_test_targets.unaryExpr([](double v) { return std::round(v); });
            test_targets = test_targets.unaryExpr([](double v) { return std::round(v); });

            vector_predicted_test_targets[j] = predicted_test_targets;
            vector_test_targets[j] = test_targets;

        }

        const size_t buf_size = 2;
        decimal accuracy = -1.0;
        decimal accuracies[buf_size];

        auto corr_predictions = Helpers::getCorrectPredictions(vector_test_targets, vector_predicted_test_targets);
        auto current_accuracy = Helpers::getAccuracy(vector_test_targets, vector_predicted_test_targets);

        // Output section
        {
            std::cout << "Epoch: " << epoch << std::endl;
            std::cout << "Correct Predictions: " << corr_predictions << " out of " << test_data_size << std::endl;
            std::cout << "Accuracy: " << current_accuracy << std::endl;
            std::cout << std::endl;
        }
        
        if (current_accuracy + decimal_eps >= 1.0) {
            patience = patience_const;
            accuracy = current_accuracy;
            nn = nn_ws;
            break;
        }

        bool is_accuracy_better = epoch == 0 || current_accuracy >= accuracy;

        if (is_accuracy_better) {
            patience = patience_const;
            accuracy = current_accuracy;
            nn = nn_ws;
            continue;
        }

        // The emergency exit/ early stopping
        --patience;
        if (patience == 0) {
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