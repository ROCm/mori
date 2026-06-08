// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
//
// CCO Device API — single include for device-side (kernel) code.
//
// Include this one header from device/kernel sources; host control-plane code
// includes cco.hpp instead. Pure umbrella: it pulls in every device-side
// facility — shared types + findWindow (cco_types.hpp), cooperative groups,
// teams, and the per-backend session classes plus their addressing helpers
// (LSA ccoGetLsaPeerPtr/ccoGetLocalPtr live in cco_lsa_impl.hpp; GDA under gda/).
#pragma once

#include "mori/cco/cco_types.hpp"

// Cooperative groups + teams used across all device sessions.
#include "mori/cco/cco_coop.hpp"
#include "mori/cco/cco_team.hpp"

// clang-format off
#include "mori/cco/cco_lsa_types.hpp"
#include "mori/cco/cco_lsa_impl.hpp"
// clang-format off

#include "mori/cco/gda/gda_device.hpp"
