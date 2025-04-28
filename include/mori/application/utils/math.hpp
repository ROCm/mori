#pragma once

#include <cmath>

namespace mori {
namespace application {

static int RoundUpPowOfTwo(int val) { return pow(2, ceil(log2(float(val)))); }

static int LogCeil2(int val) { return ceil(log2(float(val))); }

}  // namespace application
}  // namespace mori