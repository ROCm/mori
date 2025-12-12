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

#ifndef TIME_SLOT_ITEMS
#define TIME_SLOT_ITEMS \
    ITEM(SLOT_TIME_START, 0) \
    ITEM(SLOT_TIME_BEFORE_LOOP, 1) \
    ITEM(SLOT_TIME_AFTER_LOOP, 2) \
    ITEM(SLOT_TIME_BEFORE_WAIT, 3) \
    ITEM(SLOT_TIME_END, 4) \
    ITEM(SLOT_ACC_COPY_CYCLES, 5) \
    ITEM(SLOT_ACC_ATOMIC_CYCLES, 6) \
    ITEM(SLOT_ACC_PUT_CYCLES, 7) \
    ITEM(SLOT_ACC_SETUP_DURATION, 8) \
    ITEM(SLOT_ACC_TOKEN_DURATION, 9) \
    ITEM(SLOT_ACC_WEIGHT_DURATION, 10) \
    ITEM(SLOT_ACC_ITER_COUNT, 11) \
    ITEM(SLOT_ACC_ATOMIC_COUNT, 12) \
    ITEM(SLOT_ACC_PUT_COUNT, 13) \
    ITEM(SLOT_ACC_TOKEN_COUNT, 14)
#endif

