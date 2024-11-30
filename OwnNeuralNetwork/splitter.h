#pragma once

#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

struct Splitter {
public:
    Splitter() = default;

    void reset(size_t _cnt) {
        cnt = _cnt;
        resetIdcsFirst();
    }

    void pickIdcsRandomly(size_t _icdsToPick, size_t _sections = 1) {
        if (_icdsToPick > cnt) {
            return;
        }
        if (_sections > cnt) {
            return;
        }
        //auto icdsToPickPerSection = _icdsToPick / _sections;
        auto itemsPerSection = cnt / _sections;
        idcs.second.resize(0);
        for (size_t i = 0; i < _sections; ++i) {
            for (size_t j = 0; j < (_icdsToPick / _sections); ++j) {
                size_t randIndex = i * itemsPerSection + rand() % itemsPerSection;
                // printf("j: %d\trand: %u\n", j, randIndex);
                if (std::find(idcs.second.begin(), idcs.second.end(), randIndex) != idcs.second.end()) {
                    --j;
                    continue;
                }
                idcs.second.push_back(randIndex);
            }
        }
        std::sort(idcs.second.begin(), idcs.second.end());
    }

    void pickIdcsForCrossValidation(size_t _fold, size_t _sections = 1) {
        size_t m = cnt / _sections;
        size_t n = m / _fold;
        idcs.second.resize(0);
        for (size_t i = 0; i < _sections; ++i) {
            for (size_t j = 0; j < _fold; ++j) {

            }
        }

    }

    void removeIdcs() {
        if (idcs.second.size() == 0) {
            return;
        }
        resetIdcsFirst();
        for (auto it = idcs.second.cbegin(); it != idcs.second.cend(); ++it) {
            auto found = std::find(idcs.first.begin(), idcs.first.end(), *it);
            if (found == idcs.first.end()) {
                continue;
            }
            idcs.first.erase(found);
        }
    }

    void readIcdsFromFile()
    {
        // Ref.: https://stackoverflow.com/questions/15138785/how-to-read-a-file-into-vector-in-c
        // Opening the file
        std::ifstream is("file.txt", std::ios::in);
        std::istream_iterator<size_t> start(is), end;
        std::vector<size_t> numbers(start, end);
        idcs.second.resize(0);
        std::copy(numbers.begin(), numbers.end(), std::back_inserter(idcs.second));
        std::sort(idcs.second.begin(), idcs.second.end());
    }

    std::pair< std::vector<size_t>, std::vector<size_t>>& getIdcs() {
        return idcs;
    }

private:
    void resetIdcsFirst() {
        idcs.first.resize(cnt);
        std::iota(std::begin(idcs.first), std::end(idcs.first), 0);
    }

    size_t cnt = 0;
    std::pair< std::vector<size_t>, std::vector<size_t>> idcs = std::make_pair< std::vector<size_t>, std::vector<size_t>>({}, {});
};