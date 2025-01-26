#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include "nn_defs.h"

template <class T>
struct FeatureFilter {
public:
	FeatureFilter() {
        setMetaData();
	}
    void setMetaData() {
        std::vector<size_t> ws = { 0, 1, 2, 3 }; // applies for the special dataset iris.csv
        activeFeatures.clear();
        std::copy(ws.cbegin(), ws.cend(), std::back_inserter(activeFeatures));
    }

    std::vector<std::vector<std::string>> applyFilter(std::vector<std::vector<std::string>>::const_iterator _rawDataBegin, std::vector<std::vector<std::string>>::const_iterator _rawDataEnd) {
        std::vector<std::vector<std::string>> res = {};
        setMetaData();
        std::for_each(_rawDataBegin, _rawDataEnd, [&res, this](std::vector<std::string > _x) {
            std::vector<std::string> line = {};
            std::for_each(activeFeatures.cbegin(), activeFeatures.cend(), [&line, &_x](size_t _j) {
                line.push_back(_x[_j]);
                }
            );
            res.push_back(line);
            }
        );
        return res;
    }

    std::vector<std::vector<decimal>> applyFilter(std::vector<std::vector<decimal>>::const_iterator _rawDataBegin, std::vector<std::vector<decimal>>::const_iterator _rawDataEnd) {
        std::vector<std::vector<decimal>> res = {};
        setMetaData();
        std::for_each(_rawDataBegin, _rawDataEnd, [&res, this](std::vector<decimal> _x) {
            std::vector<decimal> line = {};
            std::for_each(activeFeatures.cbegin(), activeFeatures.cend(), [&line, &_x](size_t _j) {
                line.push_back(_x[_j]);
                }
            );
            res.push_back(line);
            }
        );
        return res;
    }

private:
    std::vector <size_t> activeFeatures;
};
