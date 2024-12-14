#pragma once

#include "nn_defs.h"

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
        for (auto it = targets.begin(), it1 = predicted_targets.begin(); it != targets.end(); ++it, ++it1) {
            Eigen::VectorXd rounded_it = it->unaryExpr([](double v) { return std::round(v); });
            Eigen::VectorXd rounded_it1 = it1->unaryExpr([](double v) { return std::round(v); });

            auto dotProduct = rounded_it.dot(rounded_it1);
            corr_predictions += dotProduct;
        }
        return corr_predictions;
    }

    decimal getAccuracy(const std::vector<vector_type>& targets, const std::vector<vector_type>& predicted_targets) {
        if (targets.size() != predicted_targets.size())
        {
            return -1.0;
        }

        return static_cast<decimal>(getCorrectPredictions(targets, predicted_targets)) / static_cast<decimal>(targets.size());
    }
}