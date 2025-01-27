#pragma once

#pragma once

#include <vector>
#include <cassert>
#include <iterator>
#include <string>
#include <functional>

#include "metadata.h"
#include "splitter.h"
#include "target_filter.h"
#include "feature_filter.h"

#include "nn_defs.h"
#include "helpers.h"

template<typename T>
T getTrainData(const T& _data, const std::vector<size_t>& _idcs) {
    T resData;
    resData.resize(_idcs.size());
    std::transform(_idcs.begin(), _idcs.end(), resData.begin(), [&_data](size_t j) {return _data[j]; });
    return resData;
}

namespace DataTable {
    class DataTable {

    public:
        DataTable() {
        }

        void setMetaData(const DataTableMetaData& _metaData) {
            metaData = _metaData;
        }

        void setNumericData(const std::vector<std::vector<decimal>>& _numericData) {
            numericData = _numericData;
        }

        void setTargets(const std::vector<std::string>& _targets) {
            targets = _targets;
        }

		std::vector<std::vector<decimal>> getNumericData() const {
			return numericData;
		}

		std::vector<decimal> getNumericDataColumn(size_t _columnIndex) const {
            std::vector<decimal> column;
            for (const auto& row : numericData) {
                if (_columnIndex < row.size()) {
                    column.push_back(row[_columnIndex]);
                }
                else {
                    throw std::out_of_range("Spaltenindex außerhalb des Bereichs");
                }
            }
            return column;
		}

        void setNumericDataColumn(size_t columnIndex, const std::vector<decimal>& _newColumn) {
            if (_newColumn.size() > numericData.size()) {
                throw std::out_of_range("Die neue Spalte ist größer als die aktuelle Anzahl an Zeilen");
            }

            // Setze die Werte der neuen Spalte
            for (size_t i = 0; i < _newColumn.size(); ++i) {
                numericData[i][columnIndex] = _newColumn[i];
            }
        }

		std::vector<std::string> getTargets() const {
			return targets;
		}

        void setData(const std::vector<std::vector<std::string>>& _rawData) {
            FeatureFilter<std::string> featureFilter;
            std::vector<std::vector<std::string>>::const_iterator it = _rawData.cbegin();
            std::advance(it, metaData.getFirstLineToRead());
            RawData rawData;
            rawData.setFilteredData(featureFilter.applyFilter(it, _rawData.cend()));
            numericData = rawData.transformData();
            targets = TargetFilter::applyFilter(it, _rawData.cend(), metaData.getTargetColumn());
        }

        void testTrainSplit(size_t _idcs) {
            splitter.reset(numericData.size());
            splitter.pickIdcsRandomly(_idcs, getTargetNames().size());
            splitter.removeIdcs();
        }

        std::vector<std::string> getTargetNames() {
            std::vector<std::string> res = {};
            std::for_each(targets.begin(), targets.end(), [&res](std::string _x) {if (!std::count(res.begin(), res.end(), _x)) res.push_back(_x); });
            return res;
        }

        size_t getNumberOfDatasets() const {
			return numericData.size();
        }

        DataTable getTrainDataTable(Splitter splitter) {
            DataTable res;
			res.setMetaData(metaData);
			res.setNumericData(getTrainData<std::vector <std::vector<decimal>>>(numericData, splitter.getIdcs().first));
            res.setTargets(getTrainData<std::vector<std::string>>(targets, splitter.getIdcs().first));
            return res;
        };

        DataTable getTestDataTable(Splitter splitter) {
            DataTable res;
            res.setMetaData(metaData);
            res.setNumericData(getTrainData<std::vector <std::vector<decimal>>>(numericData, splitter.getIdcs().second));
            res.setTargets(getTrainData<std::vector<std::string>>(targets, splitter.getIdcs().second));
            return res;
        };

		std::vector<size_t> getActiveFeatures() const {
			return metaData.activeFeatures;
		}

    private:
        DataTableMetaData metaData;
        Splitter splitter;
        std::vector<std::vector<decimal>> numericData;
        std::vector<std::string> targets;

        class RawData {
        public:
			RawData() {
			}

			void setFilteredData(const std::vector<std::vector<std::string>>& _filteredData) {
				filteredData = _filteredData;
			}

            std::vector<std::vector<decimal>> transformData(std::function<decimal(const std::string&)> _convFunc = Helpers::convertElement) {
                std::vector<std::vector<decimal>> res;
                for (size_t j = 0; j < filteredData.size(); ++j) {
                    std::vector<decimal> line;
                    for (size_t k = 0; k < filteredData[j].size(); ++k) {
                        line.push_back(_convFunc(filteredData[j][k]));
                    }
                    res.push_back(line);
                }
                return res;
            }

		private:
            std::vector<std::vector<std::string>> filteredData;
        };

    };
};

