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
// The VictoriaMetrics calls xgmi-netmon makes.
#pragma once

#include <string>

#include "http.h"

namespace xgmi {

bool VmHealthy(const Url& url, std::string* err);
// 1: VictoriaMetrics has xgmi series for this experiment and host, 0: none, -1: error (*err set).
int VmHasSeries(const Url& url, const std::string& experiment, const std::string& host,
                std::string* err);
// Deletes every xgmi_* series of this experiment and host.
bool VmDeleteSeries(const Url& url, const std::string& experiment, const std::string& host,
                    std::string* err);
// POSTs Prometheus text with timestamps to /api/v1/import/prometheus.
bool VmImport(const Url& url, const std::string& body, int timeout_ms, std::string* err);
// Imported historical data is not visible to cached queries until this is called.
void VmResetCache(const Url& url);

}  // namespace xgmi
