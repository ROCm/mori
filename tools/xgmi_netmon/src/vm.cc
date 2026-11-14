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
#include "vm.h"

#include "rates.h"

namespace xgmi {
namespace {

std::string Selector(const std::string& metric, const std::string& experiment,
                     const std::string& host) {
  return metric + "{experiment=\"" + EscapeLabel(experiment) + "\",host=\"" + EscapeLabel(host) +
         "\"}";
}

}  // namespace

bool VmHealthy(const Url& url, std::string* err) {
  std::string body;
  const int st = HttpRequest(url, "GET", "/health", "", 2000, 2000, &body, err);
  if (st == 200) return true;
  if (st > 0) *err = "/health returned HTTP " + std::to_string(st);
  return false;
}

int VmHasSeries(const Url& url, const std::string& experiment, const std::string& host,
                std::string* err) {
  std::string body;
  const std::string q = "/api/v1/series?limit=1&match[]=" +
                        UrlEncode(Selector("xgmi_interval_seconds", experiment, host));
  const int st = HttpRequest(url, "GET", q, "", 5000, 5000, &body, err);
  if (st != 200) {
    if (st > 0)
      *err = "/api/v1/series returned HTTP " + std::to_string(st) + ": " + body.substr(0, 200);
    return -1;
  }
  if (body.find("\"status\":\"success\"") == std::string::npos) {
    *err = "/api/v1/series: unexpected answer: " + body.substr(0, 200);
    return -1;
  }
  return body.find("\"data\":[]") == std::string::npos ? 1 : 0;
}

bool VmDeleteSeries(const Url& url, const std::string& experiment, const std::string& host,
                    std::string* err) {
  std::string body;
  const std::string sel = "{__name__=~\"xgmi_.*\",experiment=\"" + EscapeLabel(experiment) +
                          "\",host=\"" + EscapeLabel(host) + "\"}";
  const int st =
      HttpRequest(url, "POST", "/api/v1/admin/tsdb/delete_series?match[]=" + UrlEncode(sel), "",
                  30000, 30000, &body, err);
  if (st / 100 == 2) return true;
  if (st > 0)
    *err = "delete_series returned HTTP " + std::to_string(st) + ": " + body.substr(0, 200);
  return false;
}

bool VmImport(const Url& url, const std::string& body, int idle_ms, int total_ms,
              std::string* err) {
  std::string resp;
  const int st = HttpRequest(url, "POST", "/api/v1/import", body, idle_ms, total_ms, &resp, err);
  if (st / 100 == 2) return true;
  if (st > 0) *err = "import returned HTTP " + std::to_string(st) + ": " + resp.substr(0, 200);
  return false;
}

void VmResetCache(const Url& url) {
  std::string body, err;
  HttpRequest(url, "GET", "/internal/resetRollupResultCache", "", 5000, 5000, &body, &err);
}

}  // namespace xgmi
