#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono> // für Zeitmessung
#include <filesystem>
#include <Eigen/Dense>

#include "getcsvcontent.h"
#include "metadata.h"
#include "data_table.h"
// #include "grouped_data.h"

#include "nn_defs.h"
#include "helpers.h"

namespace fs = std::filesystem;

#ifdef _DEBUG
const auto execDir = fs::path("..\\Debug\\");
const auto execDirFallback = fs::path("..\\x64\\Debug\\");
#else
const auto execDir = fs::path("..\\Release\\");
const auto execDirFallback = fs::path("..\\x64\\Release\\");
#endif

const auto metaDataFile = fs::path("irisMetaData.txt");
const auto csvDataFile = fs::path("iris.csv");

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

// method print(const std::string& s)
// that prints s to the console
void print(const std::string& s) {
	std::cout << s << std::endl;
}

int main()
{
    // Start der Zeitmessung
    auto start = std::chrono::high_resolution_clock::now();

    std::filesystem::path cwd = fs::current_path();
	std::filesystem::path metaDataFileFullPath = cwd / (std::filesystem::exists(execDir / metaDataFile) ? execDir : execDirFallback) / metaDataFile;
	std::filesystem::path csvDataFileFullPath = cwd / (std::filesystem::exists(execDir / csvDataFile) ? execDir : execDirFallback) / csvDataFile;

    metaDataFileFullPath = std::filesystem::weakly_canonical(metaDataFileFullPath);
	csvDataFileFullPath = std::filesystem::weakly_canonical(csvDataFileFullPath);

    std::cout << "metaDataFileFullPath: " << metaDataFileFullPath << std::endl;
	std::cout << "csvDataFileFullPath: " << csvDataFileFullPath << std::endl;

    if (!std::filesystem::exists(metaDataFileFullPath) || !std::filesystem::exists(csvDataFileFullPath)) {
		std::cout << "Files not found" << std::endl;
		return 1;
    }

    std::vector<std::vector<std::string>> content = getCsvContent(csvDataFileFullPath.string());

    DataTableMetaData dataTableMetaData;
    dataTableMetaData.setMetaData(metaDataFileFullPath.string());

    DataTable::DataTable dataTable;
    dataTable.setMetaData(dataTableMetaData);
    dataTable.setData(content);

    Splitter splitter;
    splitter.reset(dataTable.getNumberOfDatasets());
    splitter.pickIdcsRandomly(30, dataTable.getTargetNames().size());
    splitter.removeIdcs();
    
    DataTable::DataTable trainDataTable = dataTable.getTrainDataTable(splitter);
    DataTable::DataTable testDataTable = dataTable.getTestDataTable(splitter);

	for (auto feature : dataTable.getActiveFeatures()) {
		print("Feature: " + std::to_string(feature));
        auto w = trainDataTable.getNumericDataColumn(feature);
		auto median = Helpers::getMedian(w);
		auto iqr = Helpers::getInterquartileRange(w);
        w = dataTable.getNumericDataColumn(feature);
        w = Helpers::getRobustScaling(dataTable.getNumericDataColumn(feature), median, iqr);
        dataTable.setNumericDataColumn(feature, w);
	}

    auto nn = NeuralNetwork(4, 4, 3, 0.12);
    auto nn_ws = nn;

    size_t test_data_size = testDataTable.getNumberOfDatasets();
	// std::cout << testDataTable.getNumberOfDatasets() << std::endl;

    size_t epochs = 250;

    const uint8_t patience_const = 10;
    uint8_t patience = patience_const;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t j = 0; j < trainDataTable.getNumberOfDatasets(); ++j) {
            vector_type train_inputs = Helpers::convertVectorElements(trainDataTable.getNumericData()[j]);
            vector_type train_targets = Helpers::getEncoding(trainDataTable.getTargets()[j]);
            nn_ws.train(train_inputs, train_targets);
        }

        std::vector<vector_type> vector_predicted_test_targets(test_data_size);
        std::vector<vector_type> vector_test_targets(test_data_size);      

        for (size_t j = 0; j < test_data_size; ++j) {
            vector_type test_inputs = Helpers::convertVectorElements(testDataTable.getNumericData()[j]);
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