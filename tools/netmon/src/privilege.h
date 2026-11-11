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
