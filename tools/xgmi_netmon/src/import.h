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
// `xgmi-netmon import`: writes recordings into VictoriaMetrics with the same rules as the live
// push.
//
// An experiment's host is imported as a whole. If VictoriaMetrics already has xgmi series for an
// experiment and host (e.g. from the live push), the import is refused unless --replace is given;
// --replace deletes only the xgmi_* series of that experiment and host first.
#pragma once

#include <string>
#include <vector>

namespace xgmi {

struct ImportOptions {
  std::vector<std::string> dirs;
  std::string url = "http://127.0.0.1:28428";
  std::string experiment;  // empty: the one stored in the recordings
  bool replace = false;
};

int RunImport(const ImportOptions& opt);

}  // namespace xgmi
