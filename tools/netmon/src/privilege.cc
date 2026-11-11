#include "privilege.h"

#include <grp.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>

namespace netmon {

bool ParseOwner(const std::string& s, uid_t* uid, gid_t* gid) {
  auto parts = Split(s, ':');
  if (parts.size() != 2 || parts[0].empty() || parts[1].empty()) return false;
  char* e1 = nullptr;
  char* e2 = nullptr;
  errno = 0;
  unsigned long u = strtoul(parts[0].c_str(), &e1, 10);
  unsigned long g = strtoul(parts[1].c_str(), &e2, 10);
  if (errno || *e1 || *e2 || u > 0xfffffffeul || g > 0xfffffffeul) return false;
  *uid = static_cast<uid_t>(u);
  *gid = static_cast<gid_t>(g);
  return true;
}

bool DetermineOwner(const std::string& owner_opt, const std::string& out_dir, Owner* out,
                    Status* error) {
  *error = Status::Ok();
  if (!owner_opt.empty()) {
    if (!ParseOwner(owner_opt, &out->uid, &out->gid)) {
      *error = Status::Error("--owner must be UID:GID (numeric), got " + owner_opt);
      return false;
    }
    out->source = "--owner";
    return true;
  }
  const char* su = getenv("SUDO_UID");
  const char* sg = getenv("SUDO_GID");
  if (su && sg && ParseOwner(std::string(su) + ":" + sg, &out->uid, &out->gid) && out->uid != 0) {
    out->source = "SUDO_UID";
    return true;
  }
  // Nearest existing ancestor (or the directory itself).
  std::string p = out_dir;
  if (p.empty() || p[0] != '/') {
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd))) return false;
    p = std::string(cwd) + "/" + p;
  }
  for (;;) {
    struct stat sb;
    if (stat(p.c_str(), &sb) == 0) {
      if (sb.st_uid == 0) return false;
      out->uid = sb.st_uid;
      out->gid = sb.st_gid;
      out->source = "owner of " + p;
      return true;
    }
    if (p == "/") return false;
    size_t s = p.find_last_of('/');
    p = s == 0 ? "/" : p.substr(0, s);
  }
}

Status DropPrivileges(const Owner& o) {
  if (setgroups(0, nullptr) != 0) return Status::Errno("setgroups", errno);
  if (setgid(o.gid) != 0) return Status::Errno("setgid", errno);
  if (setuid(o.uid) != 0) return Status::Errno("setuid", errno);
  if (getuid() != o.uid || geteuid() != o.uid || getgid() != o.gid || getegid() != o.gid) {
    return Status::Error("privilege drop did not take effect");
  }
  if (o.uid != 0 && setuid(0) == 0) return Status::Error("privilege drop is reversible");
  return Status::Ok();
}

std::vector<std::string> BuildSudoArgv(const std::string& self_path,
                                       const std::vector<std::string>& args, uid_t uid, gid_t gid) {
  std::vector<std::string> out = {"sudo", "--", self_path};
  bool has_owner = false;
  for (const auto& a : args) {
    if (a == "--sudo") continue;
    if (a == "--owner" || StartsWith(a, "--owner=")) has_owner = true;
    out.push_back(a);
  }
  if (!has_owner) {
    out.push_back("--owner");
    out.push_back(std::to_string(uid) + ":" + std::to_string(gid));
  }
  return out;
}

std::string SelfExePath() {
  char buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return "";
  buf[n] = 0;
  return buf;
}

}  // namespace netmon
