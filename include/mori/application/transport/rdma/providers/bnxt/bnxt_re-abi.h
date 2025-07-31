/*
 * Copyright (c) 2015-2024, Broadcom. All rights reserved.  The term
 * Broadcom refers to Broadcom Inc. and/or its subsidiaries.
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
 * Description: Uverbs ABI header file
 */

#ifndef __BNXT_RE_UVERBS_ABI_H__
#define __BNXT_RE_UVERBS_ABI_H__

#define BNXT_RE_ABI_VERSION			7
#define BNXT_RE_ABI_VERSION_UVERBS_IOCTL	8

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

struct bnxt_re_uctx_req {
	__aligned_u64 comp_mask;
};

#define BNXT_RE_CHIP_ID0_CHIP_NUM_SFT		0x00
#define BNXT_RE_CHIP_ID0_CHIP_REV_SFT		0x10
#define BNXT_RE_CHIP_ID0_CHIP_MET_SFT		0x18
struct bnxt_re_uctx_resp {
	__u32 dev_id;
	__u32 max_qp;
	__u32 pg_size;
	__u32 cqe_sz;
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
	__u32 pdid;
	__u64 comp_mask; /*FIXME: Not working if __aligned_u64 is used */
} __attribute__((packed));

struct bnxt_re_packet_pacing_caps {
	__u32 qp_rate_limit_min;
	__u32 qp_rate_limit_max; /* In kpbs */
	/* Corresponding bit will be set if qp type from
	 * 'enum ib_qp_type' is supported, e.g.
	 * supported_qpts |= 1 << IB_QPT_RC for brcm
	 */
	__u32 supported_qpts;
	__u32 reserved;
} __attribute__((packed));

struct bnxt_re_ah_resp {
	__u32 ah_id;
	__u64 comp_mask;
} __attribute__((packed));

struct bnxt_re_query_device_ex_resp {
	struct bnxt_re_packet_pacing_caps packet_pacing_caps;
} __attribute__((packed));

enum {
	BNXT_RE_COMP_MASK_CQ_REQ_CAP_DBR_RECOVERY = 0x1,
	BNXT_RE_COMP_MASK_CQ_REQ_CAP_DBR_PACING_NOTIFY = 0x2,
	BNXT_RE_COMP_MASK_CQ_REQ_HAS_HDBR_KADDR = 0x4
};

#define BNXT_RE_IS_DBR_RECOV_CQ(_req)					\
	(_req.comp_mask & BNXT_RE_COMP_MASK_CQ_REQ_CAP_DBR_RECOVERY)

#define BNXT_RE_IS_DBR_PACING_NOTIFY_CQ(_req)				\
	(_req.comp_mask & BNXT_RE_COMP_MASK_CQ_REQ_CAP_DBR_PACING_NOTIFY)

struct bnxt_re_cq_req {
	__u64 cq_va;
	__u64 cq_handle;
	__aligned_u64 comp_mask;
	__u64 cqprodva;
	__u64 cqconsva;
} __attribute__((packed));

enum bnxt_re_cq_mask {
	BNXT_RE_CQ_TOGGLE_PAGE_SUPPORT = 0x1,
	BNXT_RE_CQ_HDBR_KADDR_SUPPORT = 0x02
};

struct bnxt_re_cq_resp {
	__u32 cqid;
	__u32 tail;
	__u32 phase;
	__u32 rsvd;
	__aligned_u64 comp_mask;
	__u64 cq_toggle_mmap_key;
	__u64 hdbr_cq_mmap_key;
} __attribute__((packed));

struct bnxt_re_resize_cq_req {
	__u64 cq_va;
} __attribute__((packed));

struct bnxt_re_qp_req {
	__u64 qpsva;
	__u64 qprva;
	__u64 qp_handle;
	__u64 sqprodva;
	__u64 sqconsva;
	__u64 rqprodva;
	__u64 rqconsva;
} __attribute__((packed));

struct bnxt_re_qp_resp {
	__u32 qpid;
	__u32 hdbr_dt;
	__u64 hdbr_sq_mmap_key;
	__u64 hdbr_rq_mmap_key;
} __attribute__((packed));

enum bnxt_re_srq_mask {
	BNXT_RE_SRQ_TOGGLE_PAGE_SUPPORT = 0x1,
};

struct bnxt_re_srq_req {
	__u64 srqva;
	__u64 srq_handle;
	__u64 srqprodva;
	__u64 srqconsva;
} __attribute__((packed));

struct bnxt_re_srq_resp {
	__u32 srqid;
	__u64 hdbr_srq_mmap_key;
	__u64 srq_toggle_mmap_key;
	__aligned_u64 comp_mask;
} __attribute__((packed));

/* Modify QP */
enum {
	BNXT_RE_COMP_MASK_MQP_EX_PPP_REQ_EN_MASK = 0x1,
	BNXT_RE_COMP_MASK_MQP_EX_PPP_REQ_EN	 = 0x1,
	BNXT_RE_COMP_MASK_MQP_EX_PATH_MTU_MASK	 = 0x2
};

struct bnxt_re_modify_qp_ex_req {
	__aligned_u64 comp_mask;
	__u32 dpi;
	__u32 rsvd;
} __attribute__((packed));

struct bnxt_re_modify_qp_ex_resp {
	__aligned_u64 comp_mask;
	__u32 ppp_st_idx;
	__u32 path_mtu;
} __attribute__((packed));

enum bnxt_re_shpg_offt {
	BNXT_RE_BEG_RESV_OFFT	= 0x00,
	BNXT_RE_AVID_OFFT	= 0x10,
	BNXT_RE_AVID_SIZE	= 0x04,
	BNXT_RE_END_RESV_OFFT	= 0xFF0
};

#ifdef HAVE_UAPI_DEF

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
	BNXT_RE_ALLOC_DBR_BAR_PAGE,
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

struct bnxt_re_dv_db_region_attr {
	uint32_t dbr_handle;
	uint32_t dpi;
	uint64_t umdbr;
	void *dbr;
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

enum bnxt_re_obj_dpi_methods {
	BNXT_RE_METHOD_DBR_ALLOC = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_DBR_FREE,
	BNXT_RE_METHOD_DBR_QUERY,
};

struct bnxt_re_dv_umem {
	struct bnxt_re_dev *rdev;
	struct ib_umem *umem;
	u64 addr;
	size_t size;
	uint32_t access;
	int dmabuf_fd;
	struct bnxt_qplib_umem_info umem_info;
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

enum bnxt_re_dv_cq_methods {
	BNXT_RE_METHOD_DV_CREATE_CQ = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_DV_DESTROY_CQ,
};

enum bnxt_re_dv_destroy_cq_attrs {
	BNXT_RE_DV_DESTROY_CQ_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
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
	__u32 sq_len;   /* total len including MSN area */
	__u32 sq_slots;
	__u32 sq_wqe_sz;
	__u32 sq_psn_sz;
	__u32 sq_npsn;
	__aligned_u64 rq_va;
	__u32 rq_umem_offset;
	__u32 rq_len;
	__u32 rq_slots; /* == max_recv_wr */
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

enum bnxt_re_dv_qp_methods {
	BNXT_RE_METHOD_DV_CREATE_QP = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_METHOD_DV_DESTROY_QP,
	BNXT_RE_METHOD_DV_MODIFY_QP,
	BNXT_RE_METHOD_DV_QUERY_QP,
};

enum bnxt_re_dv_destroy_qp_attrs {
	BNXT_RE_DV_DESTROY_QP_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
};

enum bnxt_re_dv_query_qp_attrs {
	BNXT_RE_DV_QUERY_QP_HANDLE = (1U << UVERBS_ID_NS_SHIFT),
	BNXT_RE_DV_QUERY_QP_ATTR,
};
#endif

#endif
