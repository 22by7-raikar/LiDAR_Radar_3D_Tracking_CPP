#include "fusion/tracking/hungarian.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace fusion {

std::vector<int> solveAssignment(const std::vector<std::vector<double>>& cost) {
    const int rows = static_cast<int>(cost.size());
    if (rows == 0) return {};
    const int cols = static_cast<int>(cost[0].size());
    if (cols == 0) return std::vector<int>(rows, -1);
    for (const auto& row : cost)
        for (double c : row)
            if (!std::isfinite(c))
                throw std::invalid_argument("solveAssignment: costs must be finite");

    // The potential-based algorithm needs rows <= cols; transpose otherwise.
    const bool transposed = rows > cols;
    const int n = transposed ? cols : rows;
    const int m = transposed ? rows : cols;
    auto a = [&](int i, int j) { return transposed ? cost[j - 1][i - 1] : cost[i - 1][j - 1]; };

    // 1-based arrays; column 0 is a virtual start column.
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> match(m + 1, 0), way(m + 1, 0);
    for (int i = 1; i <= n; ++i) {
        match[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, inf);
        std::vector<bool> used(m + 1, false);
        // Grow an alternating tree from row i along the cheapest reduced edges
        // until it reaches a free column, then flip the augmenting path.
        do {
            used[j0] = true;
            const int i0 = match[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = a(i0, j) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[match[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (match[j0] != 0);
        do {
            const int j1 = way[j0];
            match[j0] = match[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> result(rows, -1);
    for (int j = 1; j <= m; ++j) {
        if (match[j] == 0) continue;
        if (transposed)
            result[j - 1] = match[j] - 1;
        else
            result[match[j] - 1] = j - 1;
    }
    return result;
}

}  // namespace fusion
