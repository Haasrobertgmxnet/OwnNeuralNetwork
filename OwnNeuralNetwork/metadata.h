#pragma once

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>

#include "getcsvcontent.h"

std::map<std::string, size_t> getMetaData(std::string _metaDataFile) {
    std::vector<std::vector<std::string>> rawContent = getCsvContent(_metaDataFile);
    std::map<std::string, size_t> metaData;
    std::cout << std::endl;
    std::for_each(rawContent.begin(), rawContent.end(), [&metaData](std::vector < std::string>& _x) {metaData.insert({ _x[0], std::stoul(_x[1]) }); });
    return metaData;
}

struct DataTableMetaData {
public:

    size_t getTargetColumn() const {
        return targetColumn;
    }

    size_t getFirstLineToRead() const {
        return firstLineToRead;
    }

    void setMetaData(const std::string& _file) {
        std::map<std::string, size_t> metaData = getMetaData(_file);
        targetColumn = metaData["targetColumn"];
        firstLineToRead = metaData["firstLineToRead"];
        for (size_t j = 0; j < metaData.size(); ++j) {
            std::string key = "activeFeature" + std::to_string(j);
            if (metaData.find(key) == metaData.end())
                continue;
            activeFeatures.push_back(metaData[key]);
        }
    }

    size_t targetColumn;
    size_t firstLineToRead;
    std::vector<size_t> activeFeatures;
};
