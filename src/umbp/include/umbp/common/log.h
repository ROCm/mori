// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
#pragma once

#include <cstdio>
#include <cstdlib>

#define UMBP_LOG_INFO(fmt, ...)  fprintf(stdout, "[UMBP INFO] " fmt "\n", ##__VA_ARGS__)
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
