#pragma once

#define _DOUBLE_

#ifndef _DOUBLE_
using decimal = float;
using vector_type = Eigen::VectorXf;
using matrix_type = Eigen::MatrixXf;
constexpr decimal decimal_eps = FLT_EPSILON;
#else
using decimal = double;
using vector_type = Eigen::VectorXd;
using matrix_type = Eigen::MatrixXd;
constexpr decimal decimal_eps = DBL_EPSILON;
#endif