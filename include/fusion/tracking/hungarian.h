/// @file
/// @brief Optimal assignment (Hungarian algorithm).
#pragma once

#include <vector>

namespace fusion {

/// @brief Minimum-cost assignment of rows to columns (Kuhn–Munkres).
///
/// Uses the O(n²·m) shortest-augmenting-path formulation with potentials and
/// accepts rectangular matrices. Every row is assigned when there are at least
/// as many columns as rows; otherwise every column is used and the remaining
/// rows are left unassigned.
/// @param cost Cost matrix, cost[row][col]; all rows must have the same length,
///        and every entry finite (std::invalid_argument otherwise).
/// @return Column assigned to each row, or -1 if the row is unassigned.
std::vector<int> solveAssignment(const std::vector<std::vector<double>>& cost);

}  // namespace fusion
