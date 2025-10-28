/* SPDX-License-Identifier: GPL-2.0 OR Linux-OpenIB */
/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc.  All rights reserved.
 */

#ifndef IONIC_DV_H
#define IONIC_DV_H

#include <stdbool.h>
#include <infiniband/verbs.h>
#ifdef __cplusplus
extern "C" {
#endif

struct ibv_cq;
struct ibv_qp;

/** IONIC_PD_TAG - tag used for parent domain resource allocation. */
#define IONIC_PD_TAG		((uint64_t)RDMA_DRIVER_IONIC << 32)
#define IONIC_PD_TAG_CQ		(IONIC_PD_TAG | 1)
#define IONIC_PD_TAG_SQ		(IONIC_PD_TAG | 2)
#define IONIC_PD_TAG_RQ		(IONIC_PD_TAG | 3)
#define IONIC_PD_TAG_RCQ	(IONIC_PD_TAG | 4)

/* deprecated */
#define IONIC_SQ_SIG_ALL	1
/* deprecated */
#define IONIC_SQ_SIG_HACK_HIGH	2

/** IONIC_UDMA_MASK_LOW - flag represents the udma0 pipeline in the udma mask. */
#define IONIC_UDMA_MASK_LOW	1
/** IONIC_UDMA_MASK_HIGH - flag represents the udma1 pipeline in the udma mask. */
#define IONIC_UDMA_MASK_HIGH	2

#define IONIC_DV_PUEC_NPORTS_MAX 8

/** struct ionic_dv_ctx - Context information for gpu-initiated rdma. */
struct ionic_dv_ctx {
	void			*db_page;
	uint64_t		*db_ptr;
	uint8_t			sq_qtype;
	uint8_t			rq_qtype;
	uint8_t			cq_qtype;
};

/** struct ionic_dv_ctx - Queue information for gpu-initiated rdma. */
struct ionic_dv_queue {
	void			*ptr;
	size_t			size;
	uint64_t		db_val;
	uint16_t		mask;
	uint8_t			depth_log2;
	uint8_t			stride_log2;
};

/** struct ionic_dv_ctx - CQ information for gpu-initiated rdma. */
struct ionic_dv_cq {
	struct ionic_dv_queue	q;
};

/** struct ionic_dv_ctx - QP information for gpu-initiated rdma. */
struct ionic_dv_qp {
	struct ionic_dv_queue	rq;
	struct ionic_dv_queue	sq;
};

/** struct ionic_puec_route - Info needed to setup a PUEC plane route. */
struct ionic_dv_puec_route {
        union ibv_gid	dgid;
        union ibv_gid	sgid;
        uint32_t	flow_label;
        uint8_t		hop_limit;
        uint8_t		traffic_class;
	uint8_t		sl;
	uint8_t		rsvd[5];
	uint32_t	flags;
};

/**
 * ionic_dv_is_ionic_ctx - Test if context belongs to ionic provider.
 */
bool ionic_dv_is_ionic_ctx(struct ibv_context *ibctx);

/**
 * ionic_dv_is_ionic_pd - Test if pd belongs to ionic provider.
 */
bool ionic_dv_is_ionic_pd(struct ibv_pd *ibpd);

/**
 * ionic_dv_is_ionic_cq - Test if cq belongs to ionic provider.
 */
bool ionic_dv_is_ionic_cq(struct ibv_cq *ibcq);

/**
 * ionic_dv_is_ionic_qp - Test if qp belongs to ionic provider.
 */
bool ionic_dv_is_ionic_qp(struct ibv_qp *ibqp);


/**
 * ionic_dv_ctx_get_udma_count - Get number of udma pipelines.
 */
uint8_t ionic_dv_ctx_get_udma_count(struct ibv_context *ibctx);

/**
 * ionic_dv_ctx_get_udma_mask - Get mask of udma pipeline ids.
 */
uint8_t ionic_dv_ctx_get_udma_mask(struct ibv_context *ibctx);

/**
 * ionic_dv_pd_get_udma_mask - Get mask of udma pipeline ids of pd or parent domain.
 */
uint8_t ionic_dv_pd_get_udma_mask(struct ibv_pd *ibpd);

/**
 * ionic_dv_pd_set_udma_mask - Restrict pipeline ids of pd or parent domain.
 *
 * Queues associated with this pd will be restricted to one of the pipelines enabled by
 * the mask at the time of queue creation.
 *
 * Recommended usage is to create a pd, then parent domains of that pd for each different
 * udma mask.  Set the desired udma mask on each parent domain.  Then, create queues
 * associated with the parent domain with the desired udma mask.
 *
 * Alternative usage is to create a pd, and set the desired udma mask prior to creating
 * each queue.  Changing the udma mask of the pd has no effect on previously created
 * queues.
 */
int ionic_dv_pd_set_udma_mask(struct ibv_pd *ibpd, uint8_t udma_mask);

/**
 * ionic_dv_cq_get_udma_mask - Get mask of udma pipeline ids of completion queueue.
 */
uint8_t ionic_dv_cq_get_udma_mask(struct ibv_cq *ibcq);

/**
 * ionic_dv_qp_get_udma_idx - Get udma pipeline id of queueue pair.
 */
uint8_t ionic_dv_qp_get_udma_idx(struct ibv_qp *ibqp);


/**
 * ionic_dv_pd_set_sqcmb - Specify send queue preference for controller memory bar.
 *
 * Send queues associated with this pd will use the controller memory bar according to
 * this preference at the time of queue creation.
 *
 * @enable - Allow the use of the controller memory bar.
 * @expdb - Allow the use of express doorbell optimizations.
 * @require - Require preferences to be met, no fallback.
 */
int ionic_dv_pd_set_sqcmb(struct ibv_pd *ibpd, bool enable, bool expdb, bool require);

/**
 * ionic_dv_pd_set_rqcmb - Specify receive queue preference for controller memory bar.
 *
 * See ionic_dv_pd_set_sqcmb().
 */
int ionic_dv_pd_set_rqcmb(struct ibv_pd *ibpd, bool enable, bool expdb, bool require);


/**
 * ionic_dv_qp_set_gda - Enable or disable GPU-Direct Async (GDA) mode.
 *
 * In GDA mode, when the application calls ibv_post_send() or ibv_post_recv(), the
 * provider writes WQEs in the descriptor ring without ringing the doorbell.
 *
 * To ring the doorbell, after posting the work the application should query to get the
 * doorbell data, and later write that data to the memory mapped doorbell register.
 *
 * See also: ionic_dv_get_ctx()
 * See also: ionic_dv_qp_get_send_dbell_data()
 * See also: ionic_dv_qp_get_recv_dbell_data()
 *
 * @ibqp - Set GDA mode for this queue pair.
 * @enable_send - Enable GDA mode for the send queue.
 * @enable_recv - Enable GDA mode for the recv queue.
 */
int ionic_dv_qp_set_gda(struct ibv_qp *ibqp, bool enable_send, bool enable_recv);

/**
 * ionic_dv_qp_get_send_dbell_data - Get send queue doorbell data.
 *
 * In GDA mode, when the application calls ibv_post_send() the provider writes WQEs in
 * the descriptor ring without ringing the doorbell.  The application should query the
 * doorbell data immediately after posting the work.  The application requests the
 * GPU to fill the source buffers of the data transfer with the result of computation.
 * The application requests the GPU to write the doorbell data to the memory mapped
 * doorbell register immediately when the computation is complete, triggering the data
 * transfer.
 *
 * It is important that the GPU ring the doorbell in sequential order.  If work requests
 * are posted in batches A, B, and C, with respective doorbell data, the data path must
 * not write B or C before A, and must not write C before B.  It is ok to skip writing a
 * doorbell, like writing only C, which will trigger the data transfer for all of the
 * work up to that point in the sequence.
 *
 * @ibqp - Get send doorbell data for this queue pair.
 * @dbdata - Output parameter for doorbell data.
 */
int ionic_dv_qp_get_send_dbell_data(struct ibv_qp *ibqp, uint64_t *dbdata);

/**
 * ionic_dv_qp_get_recv_dbell_data - Get recv queue doorbell data.
 *
 * In GDA mode, when the application calls ibv_post_recv() the provider writes WQEs in
 * the descriptor ring without ringing the doorbell.  After polling recv completions, the
 * application can immediately re-post the receive buffers without ringing the doorbell.
 * The application should query the doorbell data immediately after posting the buffers.
 * The application requests the GPU consume the data from the receive buffers.  The
 * application requests the GPU to write the doorbell data to the memory mapped doorbell
 * register immediately after the received data is consumed, making the buffers available
 * for the next data transfer.
 *
 * It is important that the GPU ring the doorbell in sequential order.  If work requests
 * are posted in batches A, B, and C, with respective doorbell data, the data path must
 * not write B or C before A, and must not write C before B.  It is ok to skip writing a
 * doorbell, like writing only C, which will make buffers available up to that point in
 * the sequence.
 *
 * @ibqp - Get recv doorbell data for this queue pair.
 * @dbdata - Output parameter for doorbell data.
 */
int ionic_dv_qp_get_recv_dbell_data(struct ibv_qp *ibqp, uint64_t *dbdata);


/**
 * ionic_dv_get_ctx - Extract context information for gpu-initiated rdma.
 */
int ionic_dv_get_ctx(struct ionic_dv_ctx *dvctx, struct ibv_context *ibctx);

/**
 * ionic_dv_get_cq - Extract cq information for gpu-initiated rdma.
 */
int ionic_dv_get_cq(struct ionic_dv_cq *dvcq, struct ibv_cq *ibcq, uint8_t udma_idx);

/**
 * ionic_dv_get_qp - Extract qp information for gpu-initiated rdma.
 */
int ionic_dv_get_qp(struct ionic_dv_qp *dvqp, struct ibv_qp *ibqp);

/**
 * ionic_dv_qp_set_puec_plane_route - set route info for a PUEC plane.
 */
int ionic_dv_qp_set_puec_plane_route(struct ibv_qp *ibqp, uint8_t plane_idx,
				     struct ionic_dv_puec_route *ipr);
#ifdef __cplusplus
}
#endif
#endif /* IONIC_DV_H */
