#pragma once

namespace mori {
namespace application {

enum mlx5_cap_mode {
  HCA_CAP_OPMOD_GET_MAX = 0,
  HCA_CAP_OPMOD_GET_CUR = 1,
};

enum {
  MLX5_QPC_ST_RC = 0x0,
};

enum {
  MLX5_QPC_PM_STATE_MIGRATED = 0x3,
};

enum {
  MLX5_CMD_OP_QUERY_HCA_CAP = 0x100,
  MLX5_CMD_OP_INIT_HCA = 0x102,
  MLX5_CMD_OP_TEARDOWN_HCA = 0x103,
  MLX5_CMD_OP_ENABLE_HCA = 0x104,
  MLX5_CMD_OP_QUERY_PAGES = 0x107,
  MLX5_CMD_OP_MANAGE_PAGES = 0x108,
  MLX5_CMD_OP_SET_HCA_CAP = 0x109,
  MLX5_CMD_OP_QUERY_ISSI = 0x10a,
  MLX5_CMD_OP_SET_ISSI = 0x10b,
  MLX5_CMD_OP_CREATE_MKEY = 0x200,
  MLX5_CMD_OP_DESTROY_MKEY = 0x202,
  MLX5_CMD_OP_CREATE_EQ = 0x301,
  MLX5_CMD_OP_DESTROY_EQ = 0x302,
  MLX5_CMD_OP_CREATE_CQ = 0x400,
  MLX5_CMD_OP_DESTROY_CQ = 0x401,
  MLX5_CMD_OP_CREATE_QP = 0x500,
  MLX5_CMD_OP_DESTROY_QP = 0x501,
  MLX5_CMD_OP_RST2INIT_QP = 0x502,
  MLX5_CMD_OP_INIT2RTR_QP = 0x503,
  MLX5_CMD_OP_RTR2RTS_QP = 0x504,
  MLX5_CMD_OP_RTS2RTS_QP = 0x505,
  MLX5_CMD_OP_QUERY_QP = 0x50b,
  MLX5_CMD_OP_INIT2INIT_QP = 0x50e,
  MLX5_CMD_OP_CREATE_PSV = 0x600,
  MLX5_CMD_OP_DESTROY_PSV = 0x601,
  MLX5_CMD_OP_CREATE_SRQ = 0x700,
  MLX5_CMD_OP_DESTROY_SRQ = 0x701,
  MLX5_CMD_OP_CREATE_XRC_SRQ = 0x705,
  MLX5_CMD_OP_DESTROY_XRC_SRQ = 0x706,
  MLX5_CMD_OP_CREATE_DCT = 0x710,
  MLX5_CMD_OP_DESTROY_DCT = 0x711,
  MLX5_CMD_OP_QUERY_DCT = 0x713,
  MLX5_CMD_OP_CREATE_XRQ = 0x717,
  MLX5_CMD_OP_DESTROY_XRQ = 0x718,
  MLX5_CMD_OP_QUERY_ESW_FUNCTIONS = 0x740,
  MLX5_CMD_OP_QUERY_ESW_VPORT_CONTEXT = 0x752,
  MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT = 0x754,
  MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT = 0x755,
  MLX5_CMD_OP_QUERY_ROCE_ADDRESS = 0x760,
  MLX5_CMD_OP_ALLOC_Q_COUNTER = 0x771,
  MLX5_CMD_OP_DEALLOC_Q_COUNTER = 0x772,
  MLX5_CMD_OP_CREATE_SCHEDULING_ELEMENT = 0x782,
  MLX5_CMD_OP_DESTROY_SCHEDULING_ELEMENT = 0x783,
  MLX5_CMD_OP_ALLOC_PD = 0x800,
  MLX5_CMD_OP_DEALLOC_PD = 0x801,
  MLX5_CMD_OP_ALLOC_UAR = 0x802,
  MLX5_CMD_OP_DEALLOC_UAR = 0x803,
  MLX5_CMD_OP_ACCESS_REG = 0x805,
  MLX5_CMD_OP_ATTACH_TO_MCG = 0x806,
  MLX5_CMD_OP_DETACH_FROM_MCG = 0x807,
  MLX5_CMD_OP_ALLOC_XRCD = 0x80e,
  MLX5_CMD_OP_DEALLOC_XRCD = 0x80f,
  MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN = 0x816,
  MLX5_CMD_OP_DEALLOC_TRANSPORT_DOMAIN = 0x817,
  MLX5_CMD_OP_ADD_VXLAN_UDP_DPORT = 0x827,
  MLX5_CMD_OP_DELETE_VXLAN_UDP_DPORT = 0x828,
  MLX5_CMD_OP_SET_L2_TABLE_ENTRY = 0x829,
  MLX5_CMD_OP_DELETE_L2_TABLE_ENTRY = 0x82b,
  MLX5_CMD_OP_QUERY_LAG = 0x842,
  MLX5_CMD_OP_CREATE_TIR = 0x900,
  MLX5_CMD_OP_DESTROY_TIR = 0x902,
  MLX5_CMD_OP_CREATE_SQ = 0x904,
  MLX5_CMD_OP_MODIFY_SQ = 0x905,
  MLX5_CMD_OP_DESTROY_SQ = 0x906,
  MLX5_CMD_OP_CREATE_RQ = 0x908,
  MLX5_CMD_OP_DESTROY_RQ = 0x90a,
  MLX5_CMD_OP_CREATE_RMP = 0x90c,
  MLX5_CMD_OP_DESTROY_RMP = 0x90e,
  MLX5_CMD_OP_CREATE_TIS = 0x912,
  MLX5_CMD_OP_MODIFY_TIS = 0x913,
  MLX5_CMD_OP_DESTROY_TIS = 0x914,
  MLX5_CMD_OP_QUERY_TIS = 0x915,
  MLX5_CMD_OP_CREATE_RQT = 0x916,
  MLX5_CMD_OP_DESTROY_RQT = 0x918,
  MLX5_CMD_OP_CREATE_FLOW_TABLE = 0x930,
  MLX5_CMD_OP_DESTROY_FLOW_TABLE = 0x931,
  MLX5_CMD_OP_QUERY_FLOW_TABLE = 0x932,
  MLX5_CMD_OP_CREATE_FLOW_GROUP = 0x933,
  MLX5_CMD_OP_DESTROY_FLOW_GROUP = 0x934,
  MLX5_CMD_OP_SET_FLOW_TABLE_ENTRY = 0x936,
  MLX5_CMD_OP_DELETE_FLOW_TABLE_ENTRY = 0x938,
  MLX5_CMD_OP_CREATE_FLOW_COUNTER = 0x939,
  MLX5_CMD_OP_DEALLOC_FLOW_COUNTER = 0x93a,
  MLX5_CMD_OP_ALLOC_PACKET_REFORMAT_CONTEXT = 0x93d,
  MLX5_CMD_OP_DEALLOC_PACKET_REFORMAT_CONTEXT = 0x93e,
  MLX5_CMD_OP_ALLOC_MODIFY_HEADER_CONTEXT = 0x940,
  MLX5_CMD_OP_DEALLOC_MODIFY_HEADER_CONTEXT = 0x941,
  MLX5_CMD_OP_CREATE_GENERAL_OBJECT = 0xa00,
  MLX5_CMD_OP_MODIFY_GENERAL_OBJECT = 0xa01,
  MLX5_CMD_OP_QUERY_GENERAL_OBJECT = 0xa02,
  MLX5_CMD_OP_DESTROY_GENERAL_OBJECT = 0xa03,
  MLX5_CMD_OP_CREATE_UMEM = 0xa08,
  MLX5_CMD_OP_DESTROY_UMEM = 0xa0a,
  MLX5_CMD_OP_SYNC_STEERING = 0xb00,
};

enum {
  MLX5_CQE_SIZE_64B = 0x0,
  MLX5_CQE_SIZE_128B = 0x1,
};

struct mlx5_ifc_cqc_bits {
  uint8_t status[0x4];
  uint8_t as_notify[0x1];
  uint8_t initiator_src_dct[0x1];
  uint8_t dbr_umem_valid[0x1];
  uint8_t reserved_at_7[0x1];
  uint8_t cqe_sz[0x3];
  uint8_t cc[0x1];
  uint8_t reserved_at_c[0x1];
  uint8_t scqe_break_moderation_en[0x1];
  uint8_t oi[0x1];
  uint8_t cq_period_mode[0x2];
  uint8_t cqe_comp_en[0x1];
  uint8_t mini_cqe_res_format[0x2];
  uint8_t st[0x4];
  uint8_t reserved_at_18[0x1];
  uint8_t cqe_comp_layout[0x7];
  uint8_t dbr_umem_id[0x20];
  uint8_t reserved_at_40[0x14];
  uint8_t page_offset[0x6];
  uint8_t reserved_at_5a[0x2];
  uint8_t mini_cqe_res_format_ext[0x2];
  uint8_t cq_timestamp_format[0x2];
  uint8_t reserved_at_60[0x3];
  uint8_t log_cq_size[0x5];
  uint8_t uar_page[0x18];
  uint8_t reserved_at_80[0x4];
  uint8_t cq_period[0xc];
  uint8_t cq_max_count[0x10];
  uint8_t reserved_at_a0[0x18];
  uint8_t c_eqn[0x8];
  uint8_t reserved_at_c0[0x3];
  uint8_t log_page_size[0x5];
  uint8_t reserved_at_c8[0x18];
  uint8_t reserved_at_e0[0x20];
  uint8_t reserved_at_100[0x8];
  uint8_t last_notified_index[0x18];
  uint8_t reserved_at_120[0x8];
  uint8_t last_solicit_index[0x18];
  uint8_t reserved_at_140[0x8];
  uint8_t consumer_counter[0x18];
  uint8_t reserved_at_160[0x8];
  uint8_t producer_counter[0x18];
  uint8_t local_partition_id[0xc];
  uint8_t process_id[0x14];
  uint8_t reserved_at_1A0[0x20];
  uint8_t dbr_addr[0x40];
};

struct mlx5_ifc_create_cq_in_bits {
  uint8_t opcode[0x10];
  uint8_t uid[0x10];
  uint8_t reserved_at_20[0x10];
  uint8_t op_mod[0x10];
  uint8_t reserved_at_40[0x40];
  struct mlx5_ifc_cqc_bits cq_context;
  uint8_t cq_umem_offset[0x40];
  uint8_t cq_umem_id[0x20];
  uint8_t cq_umem_valid[0x1];
  uint8_t reserved_at_2e1[0x1f];
  uint8_t reserved_at_300[0x580];
  uint8_t pas[];
};

struct mlx5_ifc_create_cq_out_bits {
  uint8_t reserved_at_0[0x40];

  uint8_t reserved_at_40[0x8];
  uint8_t cqn[0x18];

  uint8_t reserved_at_60[0x20];
};

struct mlx5_ifc_query_hca_cap_in_bits {
  uint8_t opcode[0x10];
  uint8_t reserved_at_10[0x10];

  uint8_t reserved_at_20[0x10];
  uint8_t op_mod[0x10];

  uint8_t other_function[0x1];
  uint8_t reserved_at_41[0xf];
  uint8_t function_id[0x10];

  uint8_t reserved_at_60[0x20];
};

struct mlx5_ifc_query_hca_cap_out_bits {
  uint8_t status[0x8];
  uint8_t reserved_at_8[0x18];

  uint8_t syndrome[0x20];

  uint8_t reserved_at_40[0x40];

  //   union mlx5_ifc_hca_cap_union_bits capability;
};

struct mlx5_ifc_ads_bits {
  uint8_t fl[0x1];
  uint8_t free_ar[0x1];
  uint8_t reserved_at_2[0xe];
  uint8_t pkey_index[0x10];

  uint8_t reserved_at_20[0x8];
  uint8_t grh[0x1];
  uint8_t mlid[0x7];
  uint8_t rlid[0x10];

  uint8_t ack_timeout[0x5];
  uint8_t reserved_at_45[0x3];
  uint8_t src_addr_index[0x8];
  uint8_t reserved_at_50[0x4];
  uint8_t stat_rate[0x4];
  uint8_t hop_limit[0x8];

  uint8_t reserved_at_60[0x4];
  uint8_t tclass[0x8];
  uint8_t flow_label[0x14];

  uint8_t rgid_rip[16][0x8];

  uint8_t reserved_at_100[0x4];
  uint8_t f_dscp[0x1];
  uint8_t f_ecn[0x1];
  uint8_t reserved_at_106[0x1];
  uint8_t f_eth_prio[0x1];
  uint8_t ecn[0x2];
  uint8_t dscp[0x6];
  uint8_t udp_sport[0x10];

  uint8_t dei_cfi[0x1];
  uint8_t eth_prio[0x3];
  uint8_t sl[0x4];
  uint8_t vhca_port_num[0x8];
  uint8_t rmac_47_32[0x10];

  uint8_t rmac_31_0[0x20];
};

struct mlx5_ifc_qpc_bits {
  uint8_t state[0x4];
  uint8_t lag_tx_port_affinity[0x4];
  uint8_t st[0x8];
  uint8_t reserved_at_10[0x2];
  uint8_t isolate_vl_tc[0x1];
  uint8_t pm_state[0x2];
  uint8_t reserved_at_15[0x1];
  uint8_t req_e2e_credit_mode[0x2];
  uint8_t offload_type[0x4];
  uint8_t end_padding_mode[0x2];
  uint8_t reserved_at_1e[0x2];

  uint8_t wq_signature[0x1];
  uint8_t block_lb_mc[0x1];
  uint8_t atomic_like_write_en[0x1];
  uint8_t latency_sensitive[0x1];
  uint8_t reserved_at_24[0x1];
  uint8_t drain_sigerr[0x1];
  uint8_t reserved_at_26[0x2];
  uint8_t pd[0x18];

  uint8_t mtu[0x3];
  uint8_t log_msg_max[0x5];
  uint8_t reserved_at_48[0x1];
  uint8_t log_rq_size[0x4];
  uint8_t log_rq_stride[0x3];
  uint8_t no_sq[0x1];
  uint8_t log_sq_size[0x4];
  uint8_t reserved_at_55[0x3];
  uint8_t ts_format[0x2];
  uint8_t data_in_order[0x1];
  uint8_t rlky[0x1];
  uint8_t ulp_stateless_offload_mode[0x4];

  uint8_t counter_set_id[0x8];
  uint8_t uar_page[0x18];

  uint8_t reserved_at_80[0x8];
  uint8_t user_index[0x18];

  uint8_t reserved_at_a0[0x3];
  uint8_t log_page_size[0x5];
  uint8_t remote_qpn[0x18];

  struct mlx5_ifc_ads_bits primary_address_path;

  struct mlx5_ifc_ads_bits secondary_address_path;

  uint8_t log_ack_req_freq[0x4];
  uint8_t reserved_at_384[0x4];
  uint8_t log_sra_max[0x3];
  uint8_t reserved_at_38b[0x2];
  uint8_t retry_count[0x3];
  uint8_t rnr_retry[0x3];
  uint8_t reserved_at_393[0x1];
  uint8_t fre[0x1];
  uint8_t cur_rnr_retry[0x3];
  uint8_t cur_retry_count[0x3];
  uint8_t reserved_at_39b[0x5];

  uint8_t reserved_at_3a0[0x20];

  uint8_t reserved_at_3c0[0x8];
  uint8_t next_send_psn[0x18];

  uint8_t reserved_at_3e0[0x8];
  uint8_t cqn_snd[0x18];

  uint8_t reserved_at_400[0x8];
  uint8_t deth_sqpn[0x18];

  uint8_t reserved_at_420[0x20];

  uint8_t reserved_at_440[0x8];
  uint8_t last_acked_psn[0x18];

  uint8_t reserved_at_460[0x8];
  uint8_t ssn[0x18];

  uint8_t reserved_at_480[0x8];
  uint8_t log_rra_max[0x3];
  uint8_t reserved_at_48b[0x1];
  uint8_t atomic_mode[0x4];
  uint8_t rre[0x1];
  uint8_t rwe[0x1];
  uint8_t rae[0x1];
  uint8_t reserved_at_493[0x1];
  uint8_t page_offset[0x6];
  uint8_t reserved_at_49a[0x3];
  uint8_t cd_slave_receive[0x1];
  uint8_t cd_slave_send[0x1];
  uint8_t cd_master[0x1];

  uint8_t reserved_at_4a0[0x3];
  uint8_t min_rnr_nak[0x5];
  uint8_t next_rcv_psn[0x18];

  uint8_t reserved_at_4c0[0x8];
  uint8_t xrcd[0x18];

  uint8_t reserved_at_4e0[0x8];
  uint8_t cqn_rcv[0x18];

  uint8_t dbr_addr[0x40];

  uint8_t q_key[0x20];

  uint8_t reserved_at_560[0x5];
  uint8_t rq_type[0x3];
  uint8_t srqn_rmpn_xrqn[0x18];

  uint8_t reserved_at_580[0x8];
  uint8_t rmsn[0x18];

  uint8_t hw_sq_wqebb_counter[0x10];
  uint8_t sw_sq_wqebb_counter[0x10];

  uint8_t hw_rq_counter[0x20];

  uint8_t sw_rq_counter[0x20];

  uint8_t reserved_at_600[0x20];

  uint8_t reserved_at_620[0xf];
  uint8_t cgs[0x1];
  uint8_t cs_req[0x8];
  uint8_t cs_res[0x8];

  uint8_t dc_access_key[0x40];

  uint8_t reserved_at_680[0x3];
  uint8_t dbr_umem_valid[0x1];

  uint8_t reserved_at_684[0x9c];

  uint8_t dbr_umem_id[0x20];
};

struct mlx5_ifc_create_qp_out_bits {
  uint8_t status[0x8];
  uint8_t reserved_at_8[0x18];

  uint8_t syndrome[0x20];

  uint8_t reserved_at_40[0x8];
  uint8_t qpn[0x18];

  uint8_t reserved_at_60[0x20];
};

struct mlx5_ifc_create_qp_in_bits {
  uint8_t opcode[0x10];
  uint8_t uid[0x10];

  uint8_t reserved_at_20[0x10];
  uint8_t op_mod[0x10];

  uint8_t reserved_at_40[0x40];

  uint8_t opt_param_mask[0x20];

  uint8_t reserved_at_a0[0x20];

  struct mlx5_ifc_qpc_bits qpc;

  uint8_t wq_umem_offset[0x40];
  uint8_t wq_umem_id[0x20];

  uint8_t wq_umem_valid[0x1];
  uint8_t reserved_at_861[0x1f];

  uint8_t pas[0][0x40];
};

}  // namespace application
}  // namespace mori