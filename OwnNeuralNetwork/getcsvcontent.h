#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<std::vector<std::string>> getCsvContent(std::string _csvFile, const char delimiter = ',') {
    std::vector<std::vector<std::string>> csvContent = {};
    std::vector<std::string> row;
    std::string line, word;

    std::fstream file(_csvFile, std::ios::in);
    if (!file.is_open())
    {
        std::cout << "Could not open the file\n";
        return csvContent;
    }
    int i = 0;
    while (getline(file, line))
    {
        row.clear();

        std::stringstream str(line);

        while (getline(str, word, delimiter))
            row.push_back(word);
        csvContent.push_back(row);
        ++i;
    }
    return csvContent;
}

