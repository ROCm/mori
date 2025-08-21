# Copyright © Advanced Micro Devices, Inc. All rights reserved.

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.18)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

project(dependency NONE)

include(ExternalProject)

set(MSGPACK_C_DIR "${CMAKE_SOURCE_DIR}/3rdparty/msgpack-c")

# set(MSGPACK_C_TAG "origin/cpp_master")
set(MSGPACK_C_TAG "a0b2ec09da4bd823e40fa591221713951d4ec995")

# cmake-format: off
ExternalProject_Add(msgpack-c
    GIT_REPOSITORY    https://github.com/msgpack/msgpack-c.git
    GIT_TAG           ${MSGPACK_C_TAG}
    SOURCE_DIR        ${MSGPACK_C_DIR}
    BINARY_DIR        ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)
# cmake-format: on
