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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include "mori/core/utils/utils.hpp"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace mori {

class ModuleLogger {
 public:
  enum class Level { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, CRITICAL = 5 };

  static ModuleLogger& GetInstance() {
    static ModuleLogger instance;
    return instance;
  }

 private:
  ModuleLogger() {
    // Check for global environment variable
    const char* globalEnvLevel = std::getenv("MORI_GLOBAL_LOG_LEVEL");
    if (globalEnvLevel) {
      globalLevel_ = LevelFromString(std::string(globalEnvLevel));
      globalLevelSet_ = true;
    }
    // Eagerly create every known module logger while we are still single
    // threaded: the Meyers singleton guarantees this ctor runs exactly once and
    // blocks other threads until it returns, so loggers_ is fully populated
    // before any GetLogger() call can touch the map.
    for (const char* moduleName : kKnownModules) {
      InitModuleLocked(moduleName);
    }
  }

 public:
  // Initialize a module-specific logger.
  void InitModule(const std::string& moduleName, Level level = Level::ERROR) {
    std::lock_guard<std::mutex> lock(mutex_);
    InitModuleLocked(moduleName, level);
  }

 private:
  // Body of InitModule; callers must hold mutex_ (or be the constructor, which
  // runs before the instance is observable).
  void InitModuleLocked(const std::string& moduleName, Level level = Level::ERROR) {
    // Check if logger already exists
    auto existing_logger = spdlog::get(moduleName);
    std::shared_ptr<spdlog::logger> logger;

    if (existing_logger) {
      // Use existing logger
      logger = existing_logger;
    } else {
      // spdlog::stdout_color_mt throws if another thread already registered the same name
      // between our spdlog::get() check and this call — catch and fall back to the winner.
      try {
        logger = spdlog::stdout_color_mt(moduleName);
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%P] [%n] [%^%l%$] %v");
      } catch (const spdlog::spdlog_ex&) {
        logger = spdlog::get(moduleName);
      }
      // Defensive: spdlog::get may still return null if registration was
      // dropped between throw and our second lookup. Bail out cleanly
      // instead of dereferencing a null shared_ptr below.
      if (!logger) return;
    }

    // Determine the log level priority: env var > global setting > provided level
    Level finalLevel = level;

    // Check environment variable first
    std::string envVar;
    if (moduleName == "application") {
      // Use abbreviated form for APPLICATION module
      envVar = "MORI_APP_LOG_LEVEL";
    } else {
      envVar = "MORI_" + moduleName + "_LOG_LEVEL";
      std::transform(envVar.begin(), envVar.end(), envVar.begin(), ::toupper);
    }
    const char* envLevel = std::getenv(envVar.c_str());
    if (envLevel) {
      finalLevel = LevelFromString(std::string(envLevel));
      envOverrides_[moduleName] = finalLevel;
    } else if (globalLevelSet_) {
      // Use global setting if no env var and global level is set
      finalLevel = globalLevel_;
    }
    logger->set_level(ConvertLevel(finalLevel));
    loggers_[moduleName] = logger;
  }

 public:
  // Get logger for a specific module. Never returns null: unknown modules are
  // created on demand, and if creation ever fails we log a fatal message and
  // abort(), rather than returning null and letting callers silently drop logs.
  std::shared_ptr<spdlog::logger> GetLogger(const std::string& moduleName) {
    std::lock_guard<std::mutex> lock(mutex_);
    return GetLoggerLocked(moduleName);
  }

  // Set log level for a specific module
  void SetModuleLevel(const std::string& moduleName, Level level) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Check if this module is protected by environment variable
    if (HasEnvOverrideLocked(moduleName)) {
      // Log a warning but don't change the level
      auto logger = GetLoggerLocked("application");  // Use application logger for warnings
      logger->warn(
          "Attempted to change log level for module '{}' which is controlled by environment "
          "variable. Use ForceSetModuleLevel() to override.",
          moduleName);
      return;
    }

    auto logger = GetLoggerLocked(moduleName);
    logger->set_level(ConvertLevel(level));
  }

  // Set log level for all modules (global control)
  void SetGlobalLevel(Level level) {
    std::lock_guard<std::mutex> lock(mutex_);
    globalLevel_ = level;
    globalLevelSet_ = true;

    // Apply to all existing loggers
    for (auto& [name, logger] : loggers_) {
      // Skip modules controlled by environment variables
      if (!HasEnvOverrideLocked(name)) {
        logger->set_level(ConvertLevel(level));
      }
    }
  }

  // Set log level with priority check (used internally for env vars)
  void SetModuleLevelInternal(const std::string& moduleName, Level level, bool fromEnv = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto logger = GetLoggerLocked(moduleName);
    logger->set_level(ConvertLevel(level));
    if (fromEnv) {
      envOverrides_[moduleName] = level;
    }
  }

  // Check if a module has environment override
  bool HasEnvOverride(const std::string& moduleName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return HasEnvOverrideLocked(moduleName);
  }

  // Clear environment overrides for a module
  void ClearEnvOverride(const std::string& moduleName) {
    std::lock_guard<std::mutex> lock(mutex_);
    envOverrides_.erase(moduleName);
  }

  // Force set log level (ignores env protection)
  void ForceSetModuleLevel(const std::string& moduleName, Level level) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto logger = GetLoggerLocked(moduleName);
    logger->set_level(ConvertLevel(level));
  }

  // Get current global level
  Level GetGlobalLevel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return globalLevel_;
  }

  // Check if global level is set
  bool IsGlobalLevelSet() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return globalLevelSet_;
  }

  // Clear global level setting (revert to individual module control)
  void ClearGlobalLevel() {
    std::lock_guard<std::mutex> lock(mutex_);
    globalLevelSet_ = false;
  }

  // Convert string to level
  Level LevelFromString(const std::string& strLevel) {
    std::string lower_level = strLevel;
    std::transform(lower_level.begin(), lower_level.end(), lower_level.begin(), ::tolower);

    if (lower_level == "trace") return Level::TRACE;
    if (lower_level == "debug") return Level::DEBUG;
    if (lower_level == "info") return Level::INFO;
    if (lower_level == "warn") return Level::WARN;
    if (lower_level == "error") return Level::ERROR;
    if (lower_level == "critical") return Level::CRITICAL;
    return Level::ERROR;
  }

  // Applies fn to every logger; whole loop runs under the lock so loggers_ can't
  // rehash mid-iteration.
  template <typename F>
  void ForEachLogger(F&& fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [name, lg] : loggers_) fn(name, lg);
  }

 private:
  // Body of GetLogger; callers must hold mutex_. Never returns null.
  std::shared_ptr<spdlog::logger> GetLoggerLocked(const std::string& moduleName) {
    for (int i = 0; i < 2; i++) {
      auto it = loggers_.find(moduleName);
      if (it != loggers_.end() && it->second) {
        return it->second;
      }
      InitModuleLocked(moduleName);  // Create on demand
    }
    // The logging system is fundamentally broken, so we abort loudly.
    std::fprintf(stderr,
                 "FATAL: ModuleLogger failed to resolve or create a logger for module '%s' "
                 "after retrying; aborting.\n",
                 moduleName.c_str());
    std::abort();
  }

  // Body of HasEnvOverride; callers must hold mutex_.
  bool HasEnvOverrideLocked(const std::string& moduleName) const {
    return envOverrides_.find(moduleName) != envOverrides_.end();
  }

  // Known modules created eagerly by the constructor. Kept in sync with the
  // mori::modules namespace below (this class is defined before it).
  static constexpr const char* kKnownModules[] = {"application", "io",   "shmem",  "core",
                                                  "ops",         "umbp", "metrics"};

  std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> loggers_;
  std::unordered_map<std::string, Level> envOverrides_;  // Track env variable overrides
  Level globalLevel_ = Level::ERROR;
  bool globalLevelSet_ = false;
  mutable std::mutex mutex_;

  spdlog::level::level_enum ConvertLevel(Level level) {
    switch (level) {
      case Level::TRACE:
        return spdlog::level::trace;
      case Level::DEBUG:
        return spdlog::level::debug;
      case Level::INFO:
        return spdlog::level::info;
      case Level::WARN:
        return spdlog::level::warn;
      case Level::ERROR:
        return spdlog::level::err;
      case Level::CRITICAL:
        return spdlog::level::critical;
    }
    return spdlog::level::err;
  }
};

// Module names
namespace modules {
constexpr const char* APPLICATION = "application";
constexpr const char* IO = "io";
constexpr const char* SHMEM = "shmem";
constexpr const char* CORE = "core";
constexpr const char* OPS = "ops";
constexpr const char* UMBP = "umbp";
constexpr const char* METRICS = "metrics";
}  // namespace modules

// Macro helpers
#define MORI_GET_LOGGER(module) mori::ModuleLogger::GetInstance().GetLogger(module)

// General MORI logging macros.
//
// Hot-path note: MORI_GET_LOGGER() resolves the logger by a string-hashed
// unordered_map lookup. Doing that on every call — even when the level is
// disabled and spdlog drops the message — shows up on hot paths (e.g. a
// per-batch MORI_IO_TRACE). We cache the resolved logger pointer in a
// function-local static so the map lookup happens once per call-site, ever.
// spdlog's own logger->level(...) already checks the level and skips
// formatting when disabled, and it reads the level on each call, so runtime
// level changes are still honored.
#define MORI_LOG(module, level, ...)                                 \
  do {                                                               \
    static ::spdlog::logger* _mori_log_logger =                      \
        ::mori::ModuleLogger::GetInstance().GetLogger(module).get(); \
    _mori_log_logger->level(__VA_ARGS__);                            \
  } while (0)

#define MORI_TRACE(module, ...) MORI_LOG(module, trace, __VA_ARGS__)
#define MORI_DEBUG(module, ...) MORI_LOG(module, debug, __VA_ARGS__)
#define MORI_INFO(module, ...) MORI_LOG(module, info, __VA_ARGS__)
#define MORI_WARN(module, ...) MORI_LOG(module, warn, __VA_ARGS__)
#define MORI_ERROR(module, ...) MORI_LOG(module, error, __VA_ARGS__)
#define MORI_CRITICAL(module, ...) MORI_LOG(module, critical, __VA_ARGS__)

// Module-specific logging macros
#define MORI_APP_TRACE(...) MORI_TRACE(mori::modules::APPLICATION, __VA_ARGS__)
#define MORI_APP_DEBUG(...) MORI_DEBUG(mori::modules::APPLICATION, __VA_ARGS__)
#define MORI_APP_INFO(...) MORI_INFO(mori::modules::APPLICATION, __VA_ARGS__)
#define MORI_APP_WARN(...) MORI_WARN(mori::modules::APPLICATION, __VA_ARGS__)
#define MORI_APP_ERROR(...) MORI_ERROR(mori::modules::APPLICATION, __VA_ARGS__)
#define MORI_APP_CRITICAL(...) MORI_CRITICAL(mori::modules::APPLICATION, __VA_ARGS__)

#define MORI_IO_TRACE(...) MORI_TRACE(mori::modules::IO, __VA_ARGS__)
#define MORI_IO_DEBUG(...) MORI_DEBUG(mori::modules::IO, __VA_ARGS__)
#define MORI_IO_INFO(...) MORI_INFO(mori::modules::IO, __VA_ARGS__)
#define MORI_IO_WARN(...) MORI_WARN(mori::modules::IO, __VA_ARGS__)
#define MORI_IO_ERROR(...) MORI_ERROR(mori::modules::IO, __VA_ARGS__)
#define MORI_IO_CRITICAL(...) MORI_CRITICAL(mori::modules::IO, __VA_ARGS__)

#define MORI_SHMEM_TRACE(...) MORI_TRACE(mori::modules::SHMEM, __VA_ARGS__)
#define MORI_SHMEM_DEBUG(...) MORI_DEBUG(mori::modules::SHMEM, __VA_ARGS__)
#define MORI_SHMEM_INFO(...) MORI_INFO(mori::modules::SHMEM, __VA_ARGS__)
#define MORI_SHMEM_WARN(...) MORI_WARN(mori::modules::SHMEM, __VA_ARGS__)
#define MORI_SHMEM_ERROR(...) MORI_ERROR(mori::modules::SHMEM, __VA_ARGS__)
#define MORI_SHMEM_CRITICAL(...) MORI_CRITICAL(mori::modules::SHMEM, __VA_ARGS__)

#define MORI_CORE_TRACE(...) MORI_TRACE(mori::modules::CORE, __VA_ARGS__)
#define MORI_CORE_DEBUG(...) MORI_DEBUG(mori::modules::CORE, __VA_ARGS__)
#define MORI_CORE_INFO(...) MORI_INFO(mori::modules::CORE, __VA_ARGS__)
#define MORI_CORE_WARN(...) MORI_WARN(mori::modules::CORE, __VA_ARGS__)
#define MORI_CORE_ERROR(...) MORI_ERROR(mori::modules::CORE, __VA_ARGS__)
#define MORI_CORE_CRITICAL(...) MORI_CRITICAL(mori::modules::CORE, __VA_ARGS__)

#define MORI_OPS_TRACE(...) MORI_TRACE(mori::modules::OPS, __VA_ARGS__)
#define MORI_OPS_DEBUG(...) MORI_DEBUG(mori::modules::OPS, __VA_ARGS__)
#define MORI_OPS_INFO(...) MORI_INFO(mori::modules::OPS, __VA_ARGS__)
#define MORI_OPS_WARN(...) MORI_WARN(mori::modules::OPS, __VA_ARGS__)
#define MORI_OPS_ERROR(...) MORI_ERROR(mori::modules::OPS, __VA_ARGS__)
#define MORI_OPS_CRITICAL(...) MORI_CRITICAL(mori::modules::OPS, __VA_ARGS__)

#define MORI_UMBP_TRACE(...) MORI_TRACE(mori::modules::UMBP, __VA_ARGS__)
#define MORI_UMBP_DEBUG(...) MORI_DEBUG(mori::modules::UMBP, __VA_ARGS__)
#define MORI_UMBP_INFO(...) MORI_INFO(mori::modules::UMBP, __VA_ARGS__)
#define MORI_UMBP_WARN(...) MORI_WARN(mori::modules::UMBP, __VA_ARGS__)
#define MORI_UMBP_ERROR(...) MORI_ERROR(mori::modules::UMBP, __VA_ARGS__)
#define MORI_UMBP_CRITICAL(...) MORI_CRITICAL(mori::modules::UMBP, __VA_ARGS__)

#define MORI_METRICS_TRACE(...) MORI_TRACE(mori::modules::METRICS, __VA_ARGS__)
#define MORI_METRICS_DEBUG(...) MORI_DEBUG(mori::modules::METRICS, __VA_ARGS__)
#define MORI_METRICS_INFO(...) MORI_INFO(mori::modules::METRICS, __VA_ARGS__)
#define MORI_METRICS_WARN(...) MORI_WARN(mori::modules::METRICS, __VA_ARGS__)
#define MORI_METRICS_ERROR(...) MORI_ERROR(mori::modules::METRICS, __VA_ARGS__)
#define MORI_METRICS_CRITICAL(...) MORI_CRITICAL(mori::modules::METRICS, __VA_ARGS__)

// Scoped Timer class.
//
// Hot-path note: the timer only logs at DEBUG level, so on any normal run
// (INFO/WARN/ERROR) it must be a true no-op. We resolve the logger once and
// check the level in the constructor; when disabled we skip the string copy,
// both clock reads, the per-call GetLogger() hashtable lookup, and the log
// call entirely. The macros cache the (module -> logger) lookup in a
// function-local static so the hashtable probe happens once per call-site.
class ScopedTimer {
 public:
  using Clock = std::chrono::steady_clock;

  // Fast path used by the MORI_TIMER / MORI_FUNCTION_TIMER macros: the logger
  // is pre-resolved and cached by the call-site, so nothing is allocated or
  // sampled unless DEBUG logging is actually enabled for this module. The name
  // is copied into an owned string only on the enabled path, so string_view
  // callers may pass temporaries safely.
  ScopedTimer(std::string_view name, spdlog::logger* logger) {
    if (MORI_UNLIKELY(logger != nullptr && logger->should_log(spdlog::level::debug))) {
      logger_ = logger;
      name_.assign(name.data(), name.size());
      start_ = Clock::now();
    }
  }

  // Legacy path: resolves the logger by module name. Still early-outs when the
  // level is disabled so it never copies strings or reads the clock needlessly.
  explicit ScopedTimer(const std::string& name,
                       const std::string& module = mori::modules::APPLICATION) {
    auto logger = ModuleLogger::GetInstance().GetLogger(module).get();
    if (MORI_UNLIKELY(logger != nullptr && logger->should_log(spdlog::level::debug))) {
      logger_ = logger;
      name_ = name;
      start_ = Clock::now();
    }
  }

  ~ScopedTimer() {
    if (MORI_UNLIKELY(logger_ != nullptr)) {
      auto end = Clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
      logger_->debug("ScopedTimer [{}] took {} ns", name_, duration);
    }
  }

  double ElapsedSeconds() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  spdlog::logger* logger_ = nullptr;
  std::string name_;
  Clock::time_point start_{};
};

// Cache the (module -> logger) resolution in a function-local static so the
// unordered_map<string, logger> lookup runs once per call-site instead of on
// every invocation of the timed function.
#define MORI_TIMER(name, module)                                    \
  static const std::shared_ptr<spdlog::logger> _mori_timer_logger = \
      ::mori::ModuleLogger::GetInstance().GetLogger(module);        \
  ::mori::ScopedTimer timer_instance(name, _mori_timer_logger.get())
#define MORI_FUNCTION_TIMER(module)                                 \
  static const std::shared_ptr<spdlog::logger> _mori_timer_logger = \
      ::mori::ModuleLogger::GetInstance().GetLogger(module);        \
  ::mori::ScopedTimer timer_instance(__PRETTY_FUNCTION__, _mori_timer_logger.get())

// Initialization helper functions
inline void InitializeLogging() {
  auto& logger = ModuleLogger::GetInstance();

  // Initialize all modules with default ERROR level
  logger.InitModule(modules::APPLICATION);
  logger.InitModule(modules::IO);
  logger.InitModule(modules::SHMEM);
  logger.InitModule(modules::CORE);
  logger.InitModule(modules::OPS);
  logger.InitModule(modules::UMBP);
  logger.InitModule(modules::METRICS);
}

inline void InitializeLogging(const std::string& globalLevel) {
  InitializeLogging();
  auto& logger = ModuleLogger::GetInstance();
  auto level = logger.LevelFromString(globalLevel);
  logger.SetGlobalLevel(level);
}

inline void InitializeLoggingFromEnv() {
  InitializeLogging();
  auto& logger = ModuleLogger::GetInstance();

  // Check global log level first
  const char* globalLevel = std::getenv("MORI_GLOBAL_LOG_LEVEL");
  if (globalLevel) {
    auto level = logger.LevelFromString(globalLevel);
    logger.SetGlobalLevel(level);
    MORI_INFO(modules::APPLICATION, "Set global MORI log level to {} from MORI_GLOBAL_LOG_LEVEL",
              globalLevel);
  }

  // Check module-specific log levels (these override global)
  const char* appLevel = std::getenv("MORI_APP_LOG_LEVEL");
  if (appLevel) {
    auto level = logger.LevelFromString(appLevel);
    logger.SetModuleLevelInternal(modules::APPLICATION, level, true);
    MORI_APP_INFO("Set APPLICATION log level to {} from MORI_APP_LOG_LEVEL", appLevel);
  }

  const char* ioLevel = std::getenv("MORI_IO_LOG_LEVEL");
  if (ioLevel) {
    auto level = logger.LevelFromString(ioLevel);
    logger.SetModuleLevelInternal(modules::IO, level, true);
    MORI_IO_INFO("Set IO log level to {} from MORI_IO_LOG_LEVEL", ioLevel);
  }

  const char* shmemLevel = std::getenv("MORI_SHMEM_LOG_LEVEL");
  if (shmemLevel) {
    auto level = logger.LevelFromString(shmemLevel);
    logger.SetModuleLevelInternal(modules::SHMEM, level, true);
    MORI_SHMEM_INFO("Set SHMEM log level to {} from MORI_SHMEM_LOG_LEVEL", shmemLevel);
  }

  const char* coreLevel = std::getenv("MORI_CORE_LOG_LEVEL");
  if (coreLevel) {
    auto level = logger.LevelFromString(coreLevel);
    logger.SetModuleLevelInternal(modules::CORE, level, true);
    MORI_CORE_INFO("Set CORE log level to {} from MORI_CORE_LOG_LEVEL", coreLevel);
  }

  const char* opsLevel = std::getenv("MORI_OPS_LOG_LEVEL");
  if (opsLevel) {
    auto level = logger.LevelFromString(opsLevel);
    logger.SetModuleLevelInternal(modules::OPS, level, true);
    MORI_OPS_INFO("Set OPS log level to {} from MORI_OPS_LOG_LEVEL", opsLevel);
  }

  const char* umbpLevel = std::getenv("MORI_UMBP_LOG_LEVEL");
  if (umbpLevel) {
    auto level = logger.LevelFromString(umbpLevel);
    logger.SetModuleLevelInternal(modules::UMBP, level, true);
    MORI_UMBP_INFO("Set UMBP log level to {} from MORI_UMBP_LOG_LEVEL", umbpLevel);
  }

  // Check for log pattern override
  const char* logPattern = std::getenv("MORI_LOG_PATTERN");
  if (logPattern) {
    logger.ForEachLogger([&](auto&, auto& moduleLogger) { moduleLogger->set_pattern(logPattern); });
    MORI_INFO(modules::APPLICATION, "Set custom log pattern from MORI_LOG_PATTERN: {}", logPattern);
  }

  // Check for log output file
  const char* logFile = std::getenv("MORI_LOG_FILE");
  if (logFile) {
    try {
      // Create file sink and add it to all loggers
      auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFile, true);
      logger.ForEachLogger(
          [&](auto&, auto& moduleLogger) { moduleLogger->sinks().push_back(file_sink); });
      MORI_INFO(modules::APPLICATION, "Added log file output: {}", logFile);
    } catch (const std::exception& e) {
      MORI_ERROR(modules::APPLICATION, "Failed to open log file {}: {}", logFile, e.what());
    }
  }

  // Check if logging should be disabled
  const char* disableLogging = std::getenv("MORI_DISABLE_LOGGING");
  if (disableLogging &&
      (std::string(disableLogging) == "1" || std::string(disableLogging) == "true")) {
    logger.SetGlobalLevel(ModuleLogger::Level::CRITICAL);
    // Note: We can't log this since logging is disabled!
  }
}

inline void SetModuleLogLevel(const std::string& moduleName, const std::string& level) {
  auto& logger = ModuleLogger::GetInstance();
  auto logLevel = logger.LevelFromString(level);
  logger.SetModuleLevel(moduleName, logLevel);
}

// Force set module log level (ignores env variable overrides)
inline void ForceSetModuleLogLevel(const std::string& moduleName, const std::string& level) {
  auto& logger = ModuleLogger::GetInstance();
  auto logLevel = logger.LevelFromString(level);
  logger.ClearEnvOverride(moduleName);  // Remove env override
  logger.SetModuleLevel(moduleName, logLevel);
}

// Check if a module's log level is controlled by environment variable
inline bool IsModuleControlledByEnv(const std::string& moduleName) {
  auto& logger = ModuleLogger::GetInstance();
  return logger.HasEnvOverride(moduleName);
}

// Get current log level as string
inline std::string GetModuleLogLevel(const std::string& moduleName) {
  auto& logger = ModuleLogger::GetInstance();
  auto moduleLogger = logger.GetLogger(moduleName);
  auto level = moduleLogger->level();

  switch (level) {
    case spdlog::level::trace:
      return "trace";
    case spdlog::level::debug:
      return "debug";
    case spdlog::level::info:
      return "info";
    case spdlog::level::warn:
      return "warn";
    case spdlog::level::err:
      return "error";
    case spdlog::level::critical:
      return "critical";
    default:
      return "unknown";
  }
}

// Global log level control functions
inline void SetGlobalLogLevel(const std::string& level) {
  auto& logger = ModuleLogger::GetInstance();
  auto logLevel = logger.LevelFromString(level);
  logger.SetGlobalLevel(logLevel);
}

inline std::string GetGlobalLogLevel() {
  auto& logger = ModuleLogger::GetInstance();
  auto level = logger.GetGlobalLevel();

  switch (level) {
    case ModuleLogger::Level::TRACE:
      return "trace";
    case ModuleLogger::Level::DEBUG:
      return "debug";
    case ModuleLogger::Level::INFO:
      return "info";
    case ModuleLogger::Level::WARN:
      return "warn";
    case ModuleLogger::Level::ERROR:
      return "error";
    case ModuleLogger::Level::CRITICAL:
      return "critical";
    default:
      return "unknown";
  }
}

inline bool IsGlobalLogLevelSet() {
  auto& logger = ModuleLogger::GetInstance();
  return logger.IsGlobalLevelSet();
}

inline void ClearGlobalLogLevel() {
  auto& logger = ModuleLogger::GetInstance();
  logger.ClearGlobalLevel();
}

}  // namespace mori
