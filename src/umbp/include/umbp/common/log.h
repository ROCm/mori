// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include <cstdio>
#include <cstdlib>

// Log levels: 0=INFO (verbose), 1=WARN (default), 2=ERROR
// Control via UMBP_LOG_LEVEL env var.  Default is WARN — only warnings and
// errors are printed.  Set UMBP_LOG_LEVEL=0 to see all INFO messages.
inline int UmbpLogLevel() {
    static int level = [] {
        const char* env = std::getenv("UMBP_LOG_LEVEL");
        return env ? std::atoi(env) : 1;
    }();
    return level;
}

#define UMBP_LOG_INFO(fmt, ...)                                        \
    do {                                                               \
        if (UmbpLogLevel() <= 0)                                       \
            fprintf(stdout, "[UMBP INFO] " fmt "\n", ##__VA_ARGS__);   \
    } while (0)
#define UMBP_LOG_WARN(fmt, ...)  fprintf(stderr, "[UMBP WARN] " fmt "\n", ##__VA_ARGS__)
#define UMBP_LOG_ERROR(fmt, ...) fprintf(stderr, "[UMBP ERROR] " fmt "\n", ##__VA_ARGS__)

#define UMBP_CHECK(cond, fmt, ...)                                     \
    do {                                                               \
        if (!(cond)) {                                                 \
            fprintf(stderr, "[UMBP FATAL] %s:%d: " fmt "\n",          \
                    __FILE__, __LINE__, ##__VA_ARGS__);                \
            std::abort();                                              \
        }                                                              \
    } while (0)
