#pragma once
#include <algorithm>
#include <vector>
#include <string>

struct TargetFilter {
public:
    static std::vector<std::string> applyFilter(std::vector<std::vector<std::string>>::const_iterator _rawDataBegin, std::vector<std::vector<std::string>>::const_iterator _rawDataEnd, size_t _targetColumn) {
        std::vector<std::string> res = {};
        std::for_each(_rawDataBegin, _rawDataEnd, [&res, _targetColumn](std::vector<std::string> _x) {
            res.push_back(*(_x.cbegin() + _targetColumn));
            }
        );
        return res;
    }
};
