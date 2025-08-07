/*
 * Copyright (c) 2015-2024, Broadcom. All rights reserved.  The term
 * Broadcom refers to Broadcom Limited and/or its subsidiaries.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description: Main component of the bnxt_re driver
 */

#ifndef __BNXT_RE_ABI_H__
#define __BNXT_RE_ABI_H__

#if (IBVERBS_PABI_VERSION >= 17)
#include <kern-abi.h>
#else
#include <infiniband/kern-abi.h>
#endif

#ifdef IB_USER_IOCTL_CMDS
#include <rdma/ib_user_ioctl_cmds.h>
#endif

#define true			1
#define false			0

#define BNXT_RE_ABI_VERSION			7
#define BNXT_RE_ABI_VERSION_UVERBS_IOCTL	8

/*  Cu+ max inline data */
#define BNXT_RE_MAX_INLINE_SIZE		 96
#define BNXT_RE_MAX_PPP_SIZE_VAR_WQE	208
#define BNXT_RE_MAX_WCB_SIZE_VAR_WQE	224

#ifdef HAVE_J8916_ENABLED
#define BNXT_RE_FULL_FLAG_DELTA	0x80
#else
#define BNXT_RE_FULL_FLAG_DELTA	0x00
#endif

enum bnxt_re_wr_opcode {
	BNXT_RE_WR_OPCD_SEND		= 0x00,
	BNXT_RE_WR_OPCD_SEND_IMM	= 0x01,
	BNXT_RE_WR_OPCD_SEND_INVAL	= 0x02,
	BNXT_RE_WR_OPCD_RDMA_WRITE	= 0x04,
	BNXT_RE_WR_OPCD_RDMA_WRITE_IMM	= 0x05,
	BNXT_RE_WR_OPCD_RDMA_READ	= 0x06,
	BNXT_RE_WR_OPCD_ATOMIC_CS	= 0x08,
	BNXT_RE_WR_OPCD_ATOMIC_FA	= 0x0B,
	BNXT_RE_WR_OPCD_LOC_INVAL	= 0x0C,
	BNXT_RE_WR_OPCD_BIND		= 0x0E,
	BNXT_RE_WR_OPCD_RECV		= 0x80,
	BNXT_RE_WR_OPCD_INVAL		= 0xFF
};

enum bnxt_re_wr_flags {
	BNXT_RE_WR_FLAGS_INLINE		= 0x10,
	BNXT_RE_WR_FLAGS_SE		= 0x08,
	BNXT_RE_WR_FLAGS_UC_FENCE	= 0x04,
	BNXT_RE_WR_FLAGS_RD_FENCE	= 0x02,
	BNXT_RE_WR_FLAGS_SIGNALED	= 0x01
};

#define BNXT_RE_MEMW_TYPE_2		0x02
#define BNXT_RE_MEMW_TYPE_1		0x00
enum bnxt_re_wr_bind_acc {
	BNXT_RE_WR_BIND_ACC_LWR		= 0x01,
	BNXT_RE_WR_BIND_ACC_RRD		= 0x02,
	BNXT_RE_WR_BIND_ACC_RWR		= 0x04,
	BNXT_RE_WR_BIND_ACC_RAT		= 0x08,
	BNXT_RE_WR_BIND_ACC_MWB		= 0x10,
	BNXT_RE_WR_BIND_ACC_ZBVA	= 0x01,
	BNXT_RE_WR_BIND_ACC_SHIFT	= 0x10
};

enum bnxt_re_wc_type {
	BNXT_RE_WC_TYPE_SEND		= 0x00,
	BNXT_RE_WC_TYPE_RECV_RC		= 0x01,
	BNXT_RE_WC_TYPE_RECV_UD		= 0x02,
	BNXT_RE_WC_TYPE_RECV_RAW	= 0x03,
	BNXT_RE_WC_TYPE_TERM		= 0x0E,
	BNXT_RE_WC_TYPE_COFF		= 0x0F
};

#define	BNXT_RE_WC_OPCD_RECV		0x80
enum bnxt_re_req_wc_status {
	BNXT_RE_REQ_ST_OK		= 0x00,
	BNXT_RE_REQ_ST_BAD_RESP		= 0x01,
	BNXT_RE_REQ_ST_LOC_LEN		= 0x02,
	BNXT_RE_REQ_ST_LOC_QP_OP	= 0x03,
	BNXT_RE_REQ_ST_PROT		= 0x04,
	BNXT_RE_REQ_ST_MEM_OP		= 0x05,
	BNXT_RE_REQ_ST_REM_INVAL	= 0x06,
	BNXT_RE_REQ_ST_REM_ACC		= 0x07,
	BNXT_RE_REQ_ST_REM_OP		= 0x08,
	BNXT_RE_REQ_ST_RNR_NAK_XCED	= 0x09,
	BNXT_RE_REQ_ST_TRNSP_XCED	= 0x0A,
	BNXT_RE_REQ_ST_WR_FLUSH		= 0x0B
};

enum bnxt_re_rsp_wc_status {
	BNXT_RE_RSP_ST_OK		= 0x00,
	BNXT_RE_RSP_ST_LOC_ACC		= 0x01,
	BNXT_RE_RSP_ST_LOC_LEN		= 0x02,
	BNXT_RE_RSP_ST_LOC_PROT		= 0x03,
	BNXT_RE_RSP_ST_LOC_QP_OP	= 0x04,
	BNXT_RE_RSP_ST_MEM_OP		= 0x05,
	BNXT_RE_RSP_ST_REM_INVAL	= 0x06,
	BNXT_RE_RSP_ST_WR_FLUSH		= 0x07,
	BNXT_RE_RSP_ST_HW_FLUSH		= 0x08
};

enum bnxt_re_hdr_offset {
	BNXT_RE_HDR_WT_MASK		= 0xFF,
	BNXT_RE_HDR_FLAGS_MASK		= 0xFF,
	BNXT_RE_HDR_FLAGS_SHIFT		= 0x08,
	BNXT_RE_HDR_WS_MASK		= 0xFF,
	BNXT_RE_HDR_WS_SHIFT		= 0x10
};

enum bnxt_re_db_que_type {
	BNXT_RE_QUE_TYPE_SQ		= 0x00,
	BNXT_RE_QUE_TYPE_RQ		= 0x01,
	BNXT_RE_QUE_TYPE_SRQ		= 0x02,
	BNXT_RE_QUE_TYPE_SRQ_ARM	= 0x03,
	BNXT_RE_QUE_TYPE_CQ		= 0x04,
	BNXT_RE_QUE_TYPE_CQ_ARMSE	= 0x05,
	BNXT_RE_QUE_TYPE_CQ_ARMALL	= 0x06,
	BNXT_RE_QUE_TYPE_CQ_ARMENA	= 0x07,
	BNXT_RE_QUE_TYPE_SRQ_ARMENA	= 0x08,
	BNXT_RE_QUE_TYPE_CQ_CUT_ACK	= 0x09,
	BNXT_RE_PUSH_TYPE_START		= 0x0C,
	BNXT_RE_PUSH_TYPE_END		= 0x0D,
	BNXT_RE_QUE_TYPE_NULL		= 0x0F
};

enum bnxt_re_db_mask {
	BNXT_RE_DB_INDX_MASK		= 0xFFFFFFUL,
	BNXT_RE_DB_PILO_MASK		= 0x0FFUL,
	BNXT_RE_DB_PILO_SHIFT		= 0x18,
	BNXT_RE_DB_QID_MASK		= 0xFFFFFUL,
	BNXT_RE_DB_PIHI_MASK		= 0xF00UL,
	BNXT_RE_DB_PIHI_SHIFT		= 0x0C, /* Because mask is 0xF00 */
	BNXT_RE_DB_TYP_MASK		= 0x0FUL,
	BNXT_RE_DB_TYP_SHIFT		= 0x1C,
	BNXT_RE_DB_VALID_SHIFT		= 0x1A,
	BNXT_RE_DB_EPOCH_SHIFT		= 0x18,
	BNXT_RE_DB_TOGGLE_SHIFT		= 0x19,

};

enum bnxt_re_psns_mask {
	BNXT_RE_PSNS_SPSN_MASK		= 0xFFFFFF,
	BNXT_RE_PSNS_OPCD_MASK		= 0xFF,
	BNXT_RE_PSNS_OPCD_SHIFT		= 0x18,
	BNXT_RE_PSNS_NPSN_MASK		= 0xFFFFFF,
	BNXT_RE_PSNS_FLAGS_MASK		= 0xFF,
	BNXT_RE_PSNS_FLAGS_SHIFT	= 0x18
};

enum bnxt_re_msns_mask {
	BNXT_RE_SQ_MSN_SEARCH_START_PSN_MASK	= 0xFFFFFFUL,
	BNXT_RE_SQ_MSN_SEARCH_START_PSN_SHIFT	= 0,
	BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_MASK	= 0xFFFFFF000000ULL,
	BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_SHIFT	= 0x18,
	BNXT_RE_SQ_MSN_SEARCH_START_IDX_MASK	= 0xFFFF000000000000ULL,
	BNXT_RE_SQ_MSN_SEARCH_START_IDX_SHIFT	= 0x30
};

enum bnxt_re_bcqe_mask {
	BNXT_RE_BCQE_PH_MASK		= 0x01,
	BNXT_RE_BCQE_TYPE_MASK		= 0x0F,
	BNXT_RE_BCQE_TYPE_SHIFT		= 0x01,
	BNXT_RE_BCQE_RESIZE_TOG_MASK	= 0x03,
	BNXT_RE_BCQE_RESIZE_TOG_SHIFT	= 0x05,
	BNXT_RE_BCQE_STATUS_MASK	= 0xFF,
	BNXT_RE_BCQE_STATUS_SHIFT	= 0x08,
	BNXT_RE_BCQE_FLAGS_MASK		= 0xFFFFU,
	BNXT_RE_BCQE_FLAGS_SHIFT	= 0x10,
	BNXT_RE_BCQE_RWRID_MASK		= 0xFFFFFU,
	BNXT_RE_BCQE_SRCQP_MASK		= 0xFF,
	BNXT_RE_BCQE_SRCQP_SHIFT	= 0x18
};

enum bnxt_re_rc_flags_mask {
	BNXT_RE_RC_FLAGS_SRQ_RQ_MASK	= 0x01,
	BNXT_RE_RC_FLAGS_IMM_MASK	= 0x02,
	BNXT_RE_RC_FLAGS_IMM_SHIFT	= 0x01,
	BNXT_RE_RC_FLAGS_INV_MASK	= 0x04,
	BNXT_RE_RC_FLAGS_INV_SHIFT	= 0x02,
	BNXT_RE_RC_FLAGS_RDMA_MASK	= 0x08,
	BNXT_RE_RC_FLAGS_RDMA_SHIFT	= 0x03
};

enum bnxt_re_ud_flags_mask {
	BNXT_RE_UD_FLAGS_SRQ_RQ_MASK	= 0x01,
	BNXT_RE_UD_FLAGS_SRQ_RQ_SFT	= 0x00,
	BNXT_RE_UD_FLAGS_IMM_MASK	= 0x02,
	BNXT_RE_UD_FLAGS_IMM_SFT	= 0x01,
	BNXT_RE_UD_FLAGS_IP_VER_MASK	= 0x30,
	BNXT_RE_UD_FLAGS_IP_VER_SFT	= 0x4,
	BNXT_RE_UD_FLAGS_META_MASK	= 0x3C0,
	BNXT_RE_UD_FLAGS_META_SFT	= 0x6,
	BNXT_RE_UD_FLAGS_EXT_META_MASK	= 0xC00,
	BNXT_RE_UD_FLAGS_EXT_META_SFT	= 0x10,
};

enum bnxt_re_ud_cqe_mask {
	BNXT_RE_UD_CQE_MAC_MASK		= 0xFFFFFFFFFFFFULL,
	BNXT_RE_UD_CQE_SRCQPLO_MASK	= 0xFFFF,
	BNXT_RE_UD_CQE_SRCQPLO_SHIFT	= 0x30,
	BNXT_RE_UD_CQE_LEN_MASK		= 0x3FFFU
};

enum bnxt_re_que_flags_mask {
	BNXT_RE_FLAG_EPOCH_TAIL_SHIFT	= 0x0UL,
	BNXT_RE_FLAG_EPOCH_HEAD_SHIFT	= 0x1UL,
	BNXT_RE_FLAG_EPOCH_TAIL_MASK	= 0x1UL,
	BNXT_RE_FLAG_EPOCH_HEAD_MASK	= 0x2UL,
};

enum bnxt_re_db_epoch_flag_shift {
	BNXT_RE_DB_EPOCH_TAIL_SHIFT	= BNXT_RE_DB_EPOCH_SHIFT,
	BNXT_RE_DB_EPOCH_HEAD_SHIFT	= (BNXT_RE_DB_EPOCH_SHIFT - 1)
};

enum bnxt_re_ppp_st_en_mask {
	BNXT_RE_PPP_ENABLED_MASK	= 0x1UL,
	BNXT_RE_PPP_STATE_MASK		= 0x2UL,
};

enum bnxt_re_ppp_st_shift {
	BNXT_RE_PPP_ST_SHIFT		= 0x1UL
};

struct bnxt_re_db_hdr {
	__u64 typ_qid_indx; /* typ: 4, qid:20, indx:24 */
};

#define BNXT_RE_CHIP_ID0_CHIP_NUM_SFT		0x00
#define BNXT_RE_CHIP_ID0_CHIP_REV_SFT		0x10
#define BNXT_RE_CHIP_ID0_CHIP_MET_SFT		0x18

/* TBD - Syncup done with upstream */
enum {
	BNXT_RE_COMP_MASK_UCNTX_WC_DPI_ENABLED = 0x01,
	BNXT_RE_COMP_MASK_UCNTX_DBR_PACING_ENABLED = 0x02,
	BNXT_RE_COMP_MASK_UCNTX_POW2_DISABLED = 0x04,
	BNXT_RE_COMP_MASK_UCNTX_MSN_TABLE_ENABLED = 0x08,
	BNXT_RE_COMP_MASK_UCNTX_RSVD_WQE_DISABLED = 0x10,
	BNXT_RE_COMP_MASK_UCNTX_MQP_EX_SUPPORTED = 0x20,
	BNXT_RE_COMP_MASK_UCNTX_DBR_RECOVERY_ENABLED = 0x40,
	BNXT_RE_COMP_MASK_UCNTX_SMALL_RECV_WQE_DRV_SUP = 0x80,
	BNXT_RE_COMP_MASK_UCNTX_MAX_RQ_WQES = 0x100,
};

/* TBD - check the enum list */
enum bnxt_re_req_to_drv {
	BNXT_RE_COMP_MASK_REQ_UCNTX_POW2_SUPPORT = 0x01,
	BNXT_RE_COMP_MASK_REQ_UCNTX_VAR_WQE_SUPPORT = 0x02,
	BNXT_RE_COMP_MASK_REQ_UCNTX_RSVD_WQE = 0x04,
	BNXT_RE_COMP_MASK_REQ_UCNTX_SMALL_RECV_WQE_LIB_SUP = 0x08,
};

#define BNXT_RE_STATIC_WQE_MAX_SGE		0x06
#define BNXT_RE_WQE_MODES_WQE_MODE_MASK		0x01
/* bit wise modes can be extended here. */
enum bnxt_re_modes {
	BNXT_RE_WQE_MODE_STATIC =	0x00,
	BNXT_RE_WQE_MODE_VARIABLE =	0x01
	/* Other modes can be here */
};

struct bnxt_re_uctx_req {
	struct ibv_get_context cmd;
	__aligned_u64 comp_mask;
};

struct bnxt_re_uctx_resp {
#ifdef RCP_USE_IB_UVERBS
	struct ib_uverbs_get_context_resp resp;
#else
	struct ibv_get_context_resp resp;
#endif
	__u32 dev_id;
	__u32 max_qp; /* To allocate qp-table */
	__u32 pg_size;
	__u32 cqe_size;
	__u32 max_cqd;
	__u32 chip_id0;
	__u32 chip_id1;
	__u32 modes;
	__aligned_u64 comp_mask;
	__u8 db_push_mode;
	__u32 max_rq_wqes;
	__u64 dbr_pacing_mmap_key;
	__u64 uc_db_mmap_key;
	__u64 wc_db_mmap_key;
	__u64 dbr_pacing_bar_mmap_key;
	__u32 wcdpi;
	__u32 dpi;
} __attribute__((packed));


struct bnxt_re_pd_resp {
#ifdef RCP_USE_IB_UVERBS
	struct ib_uverbs_alloc_pd_resp resp;
#else
	struct ibv_alloc_pd_resp resp;
#endif
	__u32 pdid;
	__u64 comp_mask; /*FIXME: Not working if __aligned_u64 is used */
} __attribute__((packed));

struct bnxt_re_mr_resp {
#ifdef RCP_USE_IB_UVERBS
	struct ib_uverbs_reg_mr_resp resp;
#else
	struct ibv_reg_mr_resp resp;
#endif
} __attribute__((packed));

struct bnxt_re_ah_resp {
#ifdef RCP_USE_IB_UVERBS
	struct ib_uverbs_create_ah_resp resp;
#else
	struct ibv_create_ah_resp resp;
#endif
	__u32 ah_id;
	__u64 comp_mask;
} __attribute__((packed));

#ifdef VERBS_ONLY_QUERY_DEVICE_EX_DEFINED
struct bnxt_re_packet_pacing_caps {
	__u32 qp_rate_limit_min;
	__u32 qp_rate_limit_max; /* In kpbs */
	__u32 supported_qpts;
	__u32 reserved;
} __attribute__((packed));

struct bnxt_re_query_device_ex_resp {
	struct ib_uverbs_ex_query_device_resp resp;
	struct bnxt_re_packet_pacing_caps packet_pacing_caps;
} __attribute__((packed));
#endif

enum {
	BNXT_RE_COMP_MASK_CQ_REQ_CAP_DBR_RECOVERY = 0x1,
	BNXT_RE_COMP_MASK_CQ_REQ_CAP_DBR_PACING_NOTIFY = 0x2,
	BNXT_RE_COMP_MASK_CQ_REQ_HAS_HDBR_KADDR = 0x04
};

struct bnxt_re_cq_req {
	struct ibv_create_cq cmd;
	__u64 cq_va;
	__u64 cq_handle;
	__aligned_u64 comp_mask;
	__u64 cq_prodva;
	__u64 cq_consva;
} __attribute__((packed));

enum bnxt_re_cq_mask {
	BNXT_RE_CQ_TOGGLE_PAGE_SUPPORT = 0x1,
	BNXT_RE_CQ_HDBR_KADDR_SUPPORT = 0x02
};

struct bnxt_re_cq_resp {
#ifdef RCP_USE_IB_UVERBS
	struct ib_uverbs_create_cq_resp resp;
#else
	struct ibv_create_cq_resp resp;
#endif
	__u32 cqid;
	__u32 tail;
	__u32 phase;
	__u32 rsvd;
	__aligned_u64 comp_mask;
	__u64 cq_toggle_mmap_key;
	__u64 hdbr_cq_mmap_key;
} __attribute__((packed));

struct bnxt_re_resize_cq_req {
	struct ibv_resize_cq cmd;
	__u64   cq_va;
} __attribute__((packed));

struct bnxt_re_bcqe {
	__u32 flg_st_typ_ph;
	__u32 qphi_rwrid;
} __attribute__((packed));

struct bnxt_re_req_cqe {
	__u64 qp_handle;
	__u32 con_indx; /* 16 bits valid. */
	__u32 rsvd1;
	__u64 rsvd2;
} __attribute__((packed));

struct bnxt_re_rc_cqe {
	__u32 length;
	__u32 imm_key;
	__u64 qp_handle;
	__u64 mr_handle;
} __attribute__((packed));

struct bnxt_re_ud_cqe {
	__u32 length; /* 14 bits */
	__u32 immd;
	__u64 qp_handle;
	__u64 qplo_mac; /* 16:48*/
} __attribute__((packed));

struct bnxt_re_term_cqe {
	__u64 qp_handle;
	__u32 rq_sq_cidx;
	__u32 rsvd;
	__u64 rsvd1;
} __attribute__((packed));

struct bnxt_re_cutoff_cqe {
	__u64 rsvd1;
	__u64 rsvd2;
	__u64 rsvd3;
	__u8 cqe_type_toggle;
	__u8 status;
	__u16 rsvd4;
	__u32 rsvd5;
} __attribute__((packed));

/* QP */
struct bnxt_re_qp_req {
	struct ibv_create_qp cmd;
	__u64 qpsva;
	__u64 qprva;
	__u64 qp_handle;
	__u64 sqprodva;
	__u64 sqconsva;
	__u64 rqprodva;
	__u64 rqconsva;
} __attribute__((packed));

struct bnxt_re_qp_resp {
#ifdef RCP_USE_IB_UVERBS
	struct	ib_uverbs_create_qp_resp resp;
#else
	struct	ibv_create_qp_resp resp;
#endif
	__u32 qpid;
	__u32 hdbr_dt;
	__u64 hdbr_kaddr_sq;
	__u64 hdbr_kaddr_rq;
} __attribute__((packed));

enum bnxt_re_modify_ex_mask {
	BNXT_RE_MQP_PPP_REQ_EN_MASK	= 0x1UL,
	BNXT_RE_MQP_PPP_REQ_EN		= 0x1UL,
	BNXT_RE_MQP_PATH_MTU_MASK	= 0x2UL,
	BNXT_RE_MQP_PPP_IDX_MASK	= 0x7UL,
	BNXT_RE_MQP_PPP_STATE		= 0x10UL
};

#ifdef HAVE_IBV_CMD_MODIFY_QP_EX
/* Modify QP */
struct bnxt_re_modify_ex_req {
	struct	ibv_modify_qp_ex cmd;
	__aligned_u64 comp_mask;
	__u32	dpi;
	__u32	rsvd;
};

struct bnxt_re_modify_ex_resp {
#ifdef RCP_USE_IB_UVERBS
	struct	ib_uverbs_ex_modify_qp_resp resp;
#else
	struct	ibv_modify_qp_resp_ex resp;
#endif
	__aligned_u64 comp_mask;
	__u32 ppp_st_idx;
	__u32 path_mtu;
};
#endif

union lower_shdr {
	__u64 qkey_len;
	__u64 lkey_plkey;
	__u64 rva;
};

struct bnxt_re_bsqe {
	__u32 rsv_ws_fl_wt;
	__u32 key_immd;
	union lower_shdr lhdr;
} __attribute__((packed));

struct bnxt_re_psns_ext {
	__u32 opc_spsn;
	__u32 flg_npsn;
	__u16 st_slot_idx;
	__u16 rsvd0;
	__u32 rsvd1;
} __attribute__((packed));

/* sq_msn_search (size:64b/8B) */
struct bnxt_re_msns {
	__u64  start_idx_next_psn_start_psn;
} __attribute__((packed));

struct bnxt_re_psns {
	__u32 opc_spsn;
	__u32 flg_npsn;
} __attribute__((packed));

struct bnxt_re_sge {
	__u64 pa;
	__u32 lkey;
	__u32 length;
} __attribute__((packed));

struct bnxt_re_send {
	__u32 dst_qp;
	__u32 avid;
	__u64 rsvd;
} __attribute__((packed));

struct bnxt_re_raw {
	__u32 cfa_meta;
	__u32 rsvd2;
	__u64 rsvd3;
} __attribute__((packed));

struct bnxt_re_rdma {
	__u64 rva;
	__u32 rkey;
	__u32 rsvd2;
} __attribute__((packed));

struct bnxt_re_atomic {
	__u64 swp_dt;
	__u64 cmp_dt;
} __attribute__((packed));

struct bnxt_re_inval {
	__u64 rsvd[2];
} __attribute__((packed));

struct bnxt_re_bind {
	__u64 va;
	__u64 len; /* only 40 bits are valid */
} __attribute__((packed));

struct bnxt_re_brqe {
	__u32 rsv_ws_fl_wt;
	__u32 rsvd;
	__u32 wrid;
	__u32 rsvd1;
} __attribute__((packed));

struct bnxt_re_rqe {
	__u64 rsvd[2];
} __attribute__((packed));

/* SRQ */
struct bnxt_re_srq_req {
	struct ibv_create_srq cmd;
	__u64 srqva;
	__u64 srq_handle;
	__u64 srqprodva;
	__u64 srqconsva;
} __attribute__((packed));

enum bnxt_re_srq_mask {
	BNXT_RE_SRQ_TOGGLE_PAGE_SUPPORT = 0x1,
};

struct bnxt_re_srq_resp {
#ifdef RCP_USE_IB_UVERBS
	struct ib_uverbs_create_srq_resp resp;
#else
	struct ibv_create_srq_resp resp;
#endif
	__u32 srqid;
	__u64 hdbr_srq_mmap_key;
	__u64 srq_toggle_mmap_key;
	__aligned_u64 comp_mask;
} __attribute__((packed));

struct bnxt_re_srqe {
	__u64 rsvd[2];
} __attribute__((packed));

struct bnxt_re_push_wqe {
	__u64 addr[32];
} __attribute__((packed));;

#ifdef IB_USER_IOCTL_CMDS
struct bnxt_re_dv_cq_req {
	__u32 ncqe;
	__aligned_u64 va;
	__aligned_u64 comp_mask;
} __attribute__((packed));

struct bnxt_re_dv_cq_resp {
	__u32 cqid;
	__u32 tail;
	__u32 phase;
	__u32 rsvd;
	__aligned_u64 comp_mask;
} __attribute__((packed));

enum bnxt_re_objects {
	BNXT_RE_OBJECT_ALLOC_PAGE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_OBJECT_NOTIFY_DRV,
	BNXT_RE_OBJECT_GET_TOGGLE_MEM,
	BNXT_RE_OBJECT_DBR,
	BNXT_RE_OBJECT_UMEM,
	BNXT_RE_OBJECT_DV_CQ,
	BNXT_RE_OBJECT_DV_QP,
};

enum bnxt_re_alloc_page_type {
	BNXT_RE_ALLOC_WC_PAGE = 0,
	BNXT_RE_ALLOC_DBR_PACING_BAR,
	BNXT_RE_ALLOC_DBR_PAGE,
};

enum bnxt_re_var_alloc_page_attrs {
	BNXT_RE_ALLOC_PAGE_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_ALLOC_PAGE_TYPE,
	BNXT_RE_ALLOC_PAGE_DPI,
	BNXT_RE_ALLOC_PAGE_MMAP_OFFSET,
	BNXT_RE_ALLOC_PAGE_MMAP_LENGTH,
};

enum bnxt_re_alloc_page_attrs {
	BNXT_RE_DESTROY_PAGE_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
};

enum bnxt_re_alloc_page_methods {
	BNXT_RE_METHOD_ALLOC_PAGE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_DESTROY_PAGE,
};

enum bnxt_re_notify_drv_methods {
	BNXT_RE_METHOD_NOTIFY_DRV = (1U << UVERBS_ID_NS_SHIFT),
};

/* Toggle mem */
enum bnxt_re_get_toggle_mem_type {
	BNXT_RE_CQ_TOGGLE_MEM = 0,
	BNXT_RE_SRQ_TOGGLE_MEM,
};

enum bnxt_re_var_toggle_mem_attrs {
	BNXT_RE_TOGGLE_MEM_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_TOGGLE_MEM_TYPE,
	BNXT_RE_TOGGLE_MEM_RES_ID,
	BNXT_RE_TOGGLE_MEM_MMAP_PAGE,
	BNXT_RE_TOGGLE_MEM_MMAP_OFFSET,
	BNXT_RE_TOGGLE_MEM_MMAP_LENGTH,
};

enum bnxt_re_toggle_mem_attrs {
	BNXT_RE_RELEASE_TOGGLE_MEM_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
};

enum bnxt_re_toggle_mem_methods {
	BNXT_RE_METHOD_GET_TOGGLE_MEM = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_RELEASE_TOGGLE_MEM,
};

enum bnxt_re_dv_modify_qp_type {
	BNXT_RE_DV_MODIFY_QP_UDP_SPORT = 0,
};

enum bnxt_re_var_dv_modify_qp_attrs {
	BNXT_RE_DV_MODIFY_QP_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_DV_MODIFY_QP_TYPE,
	BNXT_RE_DV_MODIFY_QP_VALUE,
	BNXT_RE_DV_MODIFY_QP_REQ,
};

enum bnxt_re_obj_dbr_alloc_attrs {
	BNXT_RE_DV_ALLOC_DBR_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_DV_ALLOC_DBR_ATTR,
	BNXT_RE_DV_ALLOC_DBR_OFFSET,
};

enum bnxt_re_obj_dbr_free_attrs {
	BNXT_RE_DV_FREE_DBR_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
};

enum bnxt_re_obj_dbr_query_attrs {
	BNXT_RE_DV_QUERY_DBR_ATTR = (1U << UVERBS_ID_NS_SHIFT),
};

enum bnxt_re_obj_dbr_methods {
	BNXT_RE_METHOD_DBR_ALLOC = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_DBR_FREE,
	BNXT_RE_METHOD_DBR_QUERY,
};

enum bnxt_re_dv_umem_reg_attrs {
	BNXT_RE_UMEM_OBJ_REG_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_UMEM_OBJ_REG_ADDR,
	BNXT_RE_UMEM_OBJ_REG_LEN,
	BNXT_RE_UMEM_OBJ_REG_ACCESS,
	BNXT_RE_UMEM_OBJ_REG_DMABUF_FD,
	BNXT_RE_UMEM_OBJ_REG_PGSZ_BITMAP,
};

enum bnxt_re_dv_umem_dereg_attrs {
	BNXT_RE_UMEM_OBJ_DEREG_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
};

enum bnxt_re_dv_umem_methods {
	BNXT_RE_METHOD_UMEM_REG = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_UMEM_DEREG,
};

enum bnxt_re_dv_create_cq_attrs {
	BNXT_RE_DV_CREATE_CQ_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_DV_CREATE_CQ_REQ,
	BNXT_RE_DV_CREATE_CQ_UMEM_HANDLE,
	BNXT_RE_DV_CREATE_CQ_UMEM_OFFSET,
	BNXT_RE_DV_CREATE_CQ_RESP,
};

enum bnxt_re_dv_destroy_cq_attrs {
	BNXT_RE_DV_DESTROY_CQ_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
};

enum bnxt_re_dv_cq_methods {
	BNXT_RE_METHOD_DV_CREATE_CQ = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_DV_DESTROY_CQ
};

struct bnxt_re_dv_create_qp_req {
	int qp_type;
	__u32 max_send_wr;
	__u32 max_recv_wr;
	__u32 max_send_sge;
	__u32 max_recv_sge;
	__u32 max_inline_data;
	__u32 pd_id;
	__aligned_u64 qp_handle;
	__aligned_u64 sq_va;
	__u32 sq_umem_offset;
	__u32 sq_len;	/* total len including MSN area */
	__u32 sq_slots;
	__u32 sq_wqe_sz;
	__u32 sq_psn_sz;
	__u32 sq_npsn;
	__aligned_u64 rq_va;
	__u32 rq_umem_offset;
	__u32 rq_len;
	__u32 rq_slots;
	__u32 rq_wqe_sz;
} __attribute__((packed));

struct bnxt_re_dv_create_qp_resp {
	__u32 qpid;
} __attribute__((packed));

enum bnxt_re_dv_create_qp_attrs {
	BNXT_RE_DV_CREATE_QP_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_DV_CREATE_QP_REQ,
	BNXT_RE_DV_CREATE_QP_SEND_CQ_HANDLE,
	BNXT_RE_DV_CREATE_QP_RECV_CQ_HANDLE,
	BNXT_RE_DV_CREATE_QP_SQ_UMEM_HANDLE,
	BNXT_RE_DV_CREATE_QP_RQ_UMEM_HANDLE,
	BNXT_RE_DV_CREATE_QP_SRQ_HANDLE,
	BNXT_RE_DV_CREATE_QP_DBR_HANDLE,
	BNXT_RE_DV_CREATE_QP_RESP
};

enum bnxt_re_dv_destroy_qp_attrs {
	BNXT_RE_DV_DESTROY_QP_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
};

enum bnxt_re_dv_query_qp_attrs {
	BNXT_RE_DV_QUERY_QP_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_DV_QUERY_QP_ATTR,
};

enum bnxt_re_dv_qp_methods {
	BNXT_RE_METHOD_DV_CREATE_QP = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_DV_DESTROY_QP,
	BNXT_RE_METHOD_DV_MODIFY_QP,
	BNXT_RE_METHOD_DV_QUERY_QP,
};

#endif
#endif
