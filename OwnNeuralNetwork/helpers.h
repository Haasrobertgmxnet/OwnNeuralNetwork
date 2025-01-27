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

	decimal convertElement(const std::string& _in) {
		return std::stod(_in);
	}

    vector_type convertVectorElements(const std::vector<decimal>& _in) {
        vector_type result(_in.size());

        for (std::size_t i = 0; i < _in.size(); ++i) {
            result[i] = _in[i];
        }

        return result;
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

	decimal getArithmeticMean(std::vector<decimal> _in) {
		decimal sum = std::accumulate(_in.begin(), _in.end(), 0.0);
		return sum / _in.size();
	}

    decimal getStandardDeviation(std::vector<decimal> _in) {
        decimal mean = getArithmeticMean(_in);
        decimal sum = std::accumulate(_in.begin(), _in.end(), 0.0, [mean](decimal _x, decimal _y) {return _x + (_y - mean) * (_y - mean); });
        decimal variance = sum / (_in.size() - 1);
        return std::sqrt(variance);
    }

	decimal getMedian(std::vector<decimal> _in) {
		std::sort(_in.begin(), _in.end());
		size_t siz = _in.size();
		if (siz % 2 == 0) {
			return (_in[siz / 2 - 1] + _in[siz / 2]) / 2.0;
		}
		return _in[siz / 2];
	}

	decimal getInterquartileRange(std::vector<decimal> _in) {
		std::sort(_in.begin(), _in.end());
		size_t siz = _in.size();
		size_t q1 = siz / 4;
		size_t q3 = siz * 3 / 4;
		return _in[q3] - _in[q1];
	}

	std::vector<decimal> getStandardScaling(std::vector<decimal> _in, decimal _mean, decimal _sd) {
        std::vector<decimal> res;
        res.reserve(_in.size());
		for (auto it = _in.cbegin(); it != _in.cend(); ++it) {
			res.push_back((*it - _mean) / _sd);
		}
		return res;
	}

	std::vector<decimal> getRobustScaling(std::vector<decimal> _in, decimal _median, decimal _iqr) {
        std::vector<decimal> res;
        res.reserve(_in.size());
		for (auto it = _in.cbegin(); it != _in.cend(); ++it) {
			res.push_back((*it - _median) / _iqr);
		}
		return res;
	}

	// namespace Helpers
}