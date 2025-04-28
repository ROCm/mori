#pragma once

enum {
  MLX5_CQ_SET_CI = 0,
  MLX5_CQ_ARM_DB = 1,
};

enum ibv_gid_type_sysfs {
  IBV_GID_TYPE_SYSFS_IB_ROCE_V1,
  IBV_GID_TYPE_SYSFS_ROCE_V2,
};
