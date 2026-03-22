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
#pragma once

#include <cerrno>
#include <string>
#include <utility>

enum class IoStatusCode : int {
  Ok = 0,
  InvalidArgument = 1,
  Unavailable = 2,
  IoError = 3,
  ShortRead = 4,
  ShortWrite = 5,
  NotSupported = 6,
  Corruption = 7,
};

class IoStatus {
 public:
  IoStatus() = default;
  IoStatus(IoStatusCode code, std::string message, int os_errno = 0)
      : code_(code), message_(std::move(message)), os_errno_(os_errno) {}

  static IoStatus Ok() { return {}; }

  static IoStatus InvalidArgument(std::string message) {
    return {IoStatusCode::InvalidArgument, std::move(message)};
  }

  static IoStatus Unavailable(std::string message, int os_errno = 0) {
    return {IoStatusCode::Unavailable, std::move(message), os_errno};
  }

  static IoStatus IoError(std::string message, int os_errno = 0) {
    return {IoStatusCode::IoError, std::move(message), os_errno};
  }

  static IoStatus ShortRead(std::string message) {
    return {IoStatusCode::ShortRead, std::move(message)};
  }

  static IoStatus ShortWrite(std::string message) {
    return {IoStatusCode::ShortWrite, std::move(message)};
  }

  static IoStatus NotSupported(std::string message) {
    return {IoStatusCode::NotSupported, std::move(message)};
  }

  static IoStatus Corruption(std::string message) {
    return {IoStatusCode::Corruption, std::move(message)};
  }

  bool ok() const { return code_ == IoStatusCode::Ok; }
  IoStatusCode code() const { return code_; }
  const std::string& message() const { return message_; }
  int os_errno() const { return os_errno_; }

  explicit operator bool() const { return ok(); }

 private:
  IoStatusCode code_ = IoStatusCode::Ok;
  std::string message_;
  int os_errno_ = 0;
};
