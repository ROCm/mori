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
// Privilege handling: who owns the output, dropping root, and the explicit --sudo re-exec.
#pragma once

#include <sys/types.h>

#include <string>
#include <vector>

#include "util.h"

namespace netmon {

struct Owner {
  uid_t uid = 0;
  gid_t gid = 0;
  std::string source;  // "--owner", "SUDO_UID", "parent directory <path>"
};

// When running as root, decide which user the output should belong to, in order:
//   --owner UID:GID  ->  SUDO_UID/SUDO_GID  ->  owner of the nearest existing ancestor of
//   out_dir (if not root). Returns false if none applies (output stays root-owned).
bool DetermineOwner(const std::string& owner_opt, const std::string& out_dir, Owner* out,
                    Status* error);

bool ParseOwner(const std::string& s, uid_t* uid, gid_t* gid);

// setgroups(0), setgid, setuid, then verify root cannot be regained. Call before any thread.
Status DropPrivileges(const Owner& owner);

// argv for `--sudo`: removes "--sudo", appends "--owner <caller uid:gid>" only if the user did
// not pass --owner, and runs this binary by absolute path (no shell).
std::vector<std::string> BuildSudoArgv(const std::string& self_path,
                                       const std::vector<std::string>& args, uid_t uid, gid_t gid);

std::string SelfExePath();

}  // namespace netmon
