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
// Minimal blocking HTTP/1.0 client (plain http only), enough for VictoriaMetrics.
// HTTP/1.0 keeps responses simple: no chunked encoding, the server closes the connection.
#pragma once

#include <string>

namespace xgmi {

struct Url {
  std::string host;
  std::string port;
  std::string prefix;  // path prefix without a trailing '/', usually empty
};

bool ParseUrl(const std::string& s, Url* out, std::string* err);
std::string UrlEncode(const std::string& s);

// Sends one request with a total deadline of `timeout_ms` (connect, send and receive).
// Returns the HTTP status, or -1 with *err set. The response body is cut at 1 MiB.
int HttpRequest(const Url& url, const std::string& method, const std::string& path_and_query,
                const std::string& body, int timeout_ms, std::string* response, std::string* err);

}  // namespace xgmi
