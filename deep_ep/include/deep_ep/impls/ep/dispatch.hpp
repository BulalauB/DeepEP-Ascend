#pragma once

#include <climits>

#include <kernel_operator.h>
#include <simt_api/device_atomic_functions.h>
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/comm/handle.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/common/reduction.hpp>
#include <deep_ep/common/simt.hpp>

namespace deep_ep {

// Dispatch 的两级布局：网络按 (source token, destination rank) 去重，落 rank-major 接收区；
// dispatch_copy_epilogue 再按该 token 的 top-k 在本卡展开为 expert-major 输入。
// SIMT persistent worker：路由/计数、slot 分配、URMA WQE 构造；Scalar/MTE：metadata 打包和本地 hidden 拷贝。
// 数据面：远端 payload 走 URMA write（token hidden 与 metadata 是一对 SGE）；本 rank 绕过网络。
// 控制面：计数与 barrier 使用 symmetric peer 地址/UBMEM。count 就绪不代表 URMA payload 已完成。
// 不是 MegaMoE 的“专家端逐 route 远程 pull”，也不包含 GMM/SwiGLU 计算。
static constexpr int kNumWQEBBsPerSQESlot = 4;

template <int kNumExperts, int kNumRanks, int kNumMaxURMAEntries, int kNumMaxLocalCopies>
struct alignas(kNumUbAlignmentBytes) DispatchUBLayout {
    static constexpr int kNumMaxSGEs = kNumMaxURMAEntries + kNumRanks * handle::kNumMaxSGEPairsPerSQE;

    // Stats
    // expert_histogram 统计有效专家 route 数（尚未按 rank 去重）；rank_histogram 统计每目标 rank 的唯一 token 数。
    // 一个 token 命中同 rank 两个专家时，专家总数 +2，而该 rank 的网络 token 数仅 +1。
    int expert_histogram[kNumExperts];
    int rank_histogram[kNumRanks];
    int rank_psum[kNumRanks];

    // SGE metadata
    int sgep_idx[kNumMaxURMAEntries];
    // 逆映射：逻辑 SGE-pair 位置 -> 本 AIV token/top-k entry；-1 是末尾 SQE 的空 pair，不生成 payload SGE。
    int sgep_to_entry[kNumMaxSGEs];
    int dst_rank_idx[kNumMaxURMAEntries];
    int dst_slot_idx[kNumMaxURMAEntries];

    // SQE metadata
    // NOTES: only the physical SQE is for networking
    // 逻辑 SQE 按 peer 分段预留，物理 SQE 用原子计数分配实际 SQ 槽；二者都不等于有效 route 数。
    int lsqe_counter, psqe_counter;
    int lsqe_base_idx[kNumRanks];

    // Local copy status
    // Scalar will do copy using this flags
    // 双向 UB 握手：SIMT 发布 local_copy_ready 后 Scalar 可用最终 local slot；
    // Scalar 排空 metadata MTE3 后发布 metadata_ready，SIMT 才把本地 metadata 写到接收区。
    // 本次调用的阶段边界（接收侧不可混淆）：
    //   local_copy_ready : slot 计算完成，允许 Scalar 本地 copy；尚未保证 WQE/远端 payload 完成。
    //   metadata_ready   : metadata_send_buffer 已完成 GM 写，允许 SIMT 读取它；不等于网络完成。
    //   VF V->S wait     : SIMT WQE/count/local metadata 已结束，允许 Scalar 敲 doorbell。
    //   epilogue barrier: 本地 MTE3 + jetty CQ drain + 全 rank 到达，允许读接收 token。
    volatile int local_copy_ready, metadata_ready;
    int local_copy_dst_slot_idx[kNumMaxLocalCopies];

    // Jetty and peer metadata
    handle::HcommJettyInfo jetty_info;
    handle::HcommPeerInfo peer_info[kNumRanks];
};

template <typename ub_layout_t, int kNumHiddenBytes, int kNumSFPacks, int kNumTopk,
          int kNumMaxTokensPerVecCore, int kNumUbBytes, bool kCachedMode>
struct UBLayoutWithBuffer {
    static constexpr int kNumMetadataTokenBytes = layout::TokenLayout(0, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, not kCachedMode).get_num_bytes();
    static constexpr int kNumHiddenMTEStageBytes = math::align(kNumHiddenBytes, kNumUbAlignmentBytes);
    // 先为路由 scratch 和至少一份 hidden 槽保留空间；余量分配 metadata batch 槽，然后再计算 hidden 槽数。
    // 两个流水阶段先后使用相同事件号，但结束 metadata 时必须 drain，才能启动 hidden 的槽复用。
    static constexpr int kNumAvailableMetadataBytes = kNumUbBytes - static_cast<int>(sizeof(ub_layout_t)) - kNumHiddenMTEStageBytes;
    EP_STATIC_ASSERT(kNumAvailableMetadataBytes >= 4 * kNumMetadataTokenBytes, "Insufficient dispatch UB buffer space");
    static constexpr int kNumTokensPerMetadataMTE = kNumAvailableMetadataBytes / kNumMetadataTokenBytes >= 64 ? 64 : 4;
    static constexpr int kNumMetadataMTEStageBytes = kNumTokensPerMetadataMTE * kNumMetadataTokenBytes;
    static constexpr int kNumMetadataMTEStages = math::min(
        math::min(8, math::ceil_div(kNumMaxTokensPerVecCore, kNumTokensPerMetadataMTE)),
        kNumAvailableMetadataBytes / kNumMetadataMTEStageBytes);
    static constexpr int kNumHiddenMTEStages = math::min(8,
        (kNumUbBytes - static_cast<int>(sizeof(ub_layout_t)) - kNumMetadataMTEStages * kNumMetadataMTEStageBytes) /
            kNumHiddenMTEStageBytes);

    ub_layout_t workspace;
    alignas(kNumUbAlignmentBytes) uint8_t metadata[kNumMetadataMTEStages][kNumMetadataMTEStageBytes];
    alignas(kNumUbAlignmentBytes) uint8_t hidden[kNumHiddenMTEStages][kNumHiddenMTEStageBytes];
};

template <int kNumMaxTokensPerRank,
          int kNumRanks, int kNumExperts, int kNumTopk, int kExpertAlignment,
          bool kCachedMode, bool kDoCPUSync, int kNumVecCores, int kNumThreads,
          int kNumSGEPairsPerSQE,
          int64_t kNumTimeoutCycles,
          int kNumExpertsPerRank = kNumExperts / kNumRanks,
          int kNumMaxTokensPerVecCore = math::ceil_div(kNumMaxTokensPerRank, kNumVecCores),
          int kNumWarps = kNumThreads / warpSize,
          typename ub_layout_t = DispatchUBLayout<kNumExperts, kNumRanks, kNumMaxTokensPerVecCore * kNumTopk, kNumMaxTokensPerVecCore>>
__simt_vf__ __aicore__ __launch_bounds__(kNumThreads) void simt_persistent_worker(
    const layout::EPWorkspaceLayout workspace_layout, const layout::EPWorkspaceLayout host_workspace_layout,
    __ubuf__ ub_layout_t* ub_layout,
    __gm__ const uint8_t* x, __gm__ const uint8_t* metadata_send_buffer,
    __gm__ const uint64_t* jetty_ptrs,
    __gm__ const int64_t* topk_idx,
    __gm__ int* cumulative_local_expert_recv_stats,
    __gm__ int* psum_num_recv_tokens_per_rank,
    __gm__ int* psum_num_recv_tokens_per_expert, __gm__ int* num_unaligned_recv_tokens_per_expert,
    __gm__ int* dst_buffer_slot_idx, __gm__ int* dst_gsge_idx,
    const uint64_t recv_buffer_offset,
    const int num_hidden_bytes, const int num_metadata_bytes, const int num_token_bytes,
    const int num_tokens, const int vec_core_idx, const int rank_idx) {
    // Indices
    // TODO: try intrinsics
    // TODO: reshuffling, but also ensure coalescing
    // TODO: after shuffling, please note that the data access coalescing will be break
    // `thread_idx` can be `threadIdx.x` to save registers (without shuffling)
    const int thread_idx = static_cast<int>(threadIdx.x);
    const int warp_idx = thread_idx / warpSize;
    const int lane_idx = thread_idx % warpSize;

    // Checks
    EP_STATIC_ASSERT(kNumExperts <= kNumThreads);
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads);
    EP_STATIC_ASSERT(kNumVecCores > 0);
    EP_STATIC_ASSERT(kNumTopk > 0 and kNumTopk <= warpSize);

    // Clean: < 1 us
    if (thread_idx == 0) {
        ub_layout->lsqe_counter = 0;
        ub_layout->psqe_counter = 0;
    }
    if constexpr (not kCachedMode) {
        if (thread_idx < kNumExperts)
            ub_layout->expert_histogram[thread_idx] = 0;
        if (thread_idx < kNumRanks)
            ub_layout->rank_histogram[thread_idx] = 0;
    }
    for (int i = thread_idx; i < kNumMaxTokensPerVecCore; i += kNumThreads)
        ub_layout->local_copy_dst_slot_idx[i] = -1;
    for (int i = thread_idx; i < ub_layout_t::kNumMaxSGEs; i += kNumThreads)
        ub_layout->sgep_to_entry[i] = -1;

    asc_syncthreads();

    // Each AIV owns a bounded contiguous token range, including the final partial range.
    // cached 模式复用首次 dispatch 的 rank 去重结果、接收 slot 与 SGE 逆映射；新的 x/scale/weight 仍要传输。
    // 这是路由布局缓存，不能当成“缓存了上一轮 token payload”；调用方必须维持 handle 对应的路由。
    if constexpr (kCachedMode) {
        // Restore the cached slots and inverse SGE mapping for this vector core.
        int max_gsge_idx = -1;
        const auto num_entries = math::max(0, math::min(kNumMaxTokensPerVecCore, num_tokens - vec_core_idx * kNumMaxTokensPerVecCore)) * kNumTopk;
        for (int entry_idx = thread_idx; entry_idx < num_entries; entry_idx += kNumThreads) {
            const auto token_idx = vec_core_idx * kNumMaxTokensPerVecCore + entry_idx / kNumTopk;
            const auto topk_offset = token_idx * kNumTopk + entry_idx % kNumTopk;
            const auto dst_slot_idx = dst_buffer_slot_idx[topk_offset];
            const auto dst_rank_idx = dst_slot_idx >= 0 ? static_cast<int>(topk_idx[topk_offset]) / kNumExpertsPerRank : -1;
            const auto gsge_idx = dst_gsge_idx[topk_offset];
            ub_layout->dst_rank_idx[entry_idx] = dst_rank_idx;
            ub_layout->dst_slot_idx[entry_idx] = dst_slot_idx;
            if (gsge_idx >= 0) {
                ub_layout->sgep_to_entry[gsge_idx] = entry_idx;
                max_gsge_idx = math::max(max_gsge_idx, gsge_idx);
            } else if (dst_rank_idx == rank_idx) {
                ub_layout->local_copy_dst_slot_idx[entry_idx / kNumTopk] = dst_slot_idx;
            }
        }
        max_gsge_idx = asc_reduce_max(max_gsge_idx);
        if (lane_idx == 0)
            asc_atomic_max(&ub_layout->lsqe_counter, math::ceil_div(max_gsge_idx + 1, kNumSGEPairsPerSQE));
    } else {
        // If topk is small, a single warp can scan multiple tokens
        // 取 >=topk 的最小二次幂作 token 子组宽度，warp 内可处理多个 token；补齐 lane 的 expert=-1。
        // routing_group_mask 严格限定本 token，防止相同 rank 在相邻 token 之间被错误去重。
        static constexpr int num_threads_per_group = kNumTopk == 1 ? 1 : 1 << (32 - __builtin_clz(static_cast<unsigned>(kNumTopk - 1)));
        static constexpr int num_groups_per_wave = kNumThreads / num_threads_per_group;
        const int group_idx = thread_idx / num_threads_per_group;
        const int group_lane_idx = thread_idx % num_threads_per_group;
        const auto routing_group_mask =
            (0xffffffffu >> (warpSize - num_threads_per_group))
            << (lane_idx / num_threads_per_group * num_threads_per_group);
        // Histogram by atomic add
        // Also assign SQEs and SGEs
        // NOTES: this part cost ~6 us (with 4K tokens)
        // TODO: if local copy can be avoided, the iteration pattern will be not like this
        const auto num_iterations = math::align(math::max(0, math::min(kNumMaxTokensPerVecCore, num_tokens - vec_core_idx * kNumMaxTokensPerVecCore)), num_groups_per_wave);
        for (int i = group_idx; i < num_iterations; i += num_groups_per_wave) {
            const auto token_idx = vec_core_idx * kNumMaxTokensPerVecCore + i;
            const auto entry_idx = i * kNumTopk + group_lane_idx;

            if (i < kNumMaxTokensPerVecCore and token_idx < num_tokens) {
                // Load expert indices and atomic add on expert histogram
                int dst_expert_idx = -1;
                if (group_lane_idx < kNumTopk)
                    dst_expert_idx = static_cast<int>(topk_idx[token_idx * kNumTopk + group_lane_idx]);
                // 先统计专家 route：每个 top-k 槽保留独立专家归属；rank 去重仅减少 token 的网络副本。
                if (dst_expert_idx >= 0)
                    asc_atomic_add(ub_layout->expert_histogram + dst_expert_idx, 1);

                // Deduplicate into rank indices, allocate SGEs and store rank indices
                auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
                // 对目标 rank 去重，保留同 token/rank 的最高 top-k lane；其他 lane 写 -1。
                // 例如 token 路由专家 e0/e1 都在 rank2，仅一个 lane 为 rank2 分配 slot/SGE；
                // 发送 metadata 仍包含完整 top-k，rank2 可恢复两份专家输入。
                dst_rank_idx = simt::deduplicate<kNumRanks>(dst_rank_idx, lane_idx, routing_group_mask) ? dst_rank_idx : -1;
                if (dst_rank_idx >= 0) {
                    // The results are also the SGE indices
                    // UB 原子计数给当前 AIV 发往该 peer 的唯一 token 编号；local peer 则记录 local_copy slot，完全不占 SQ。
                    const auto dst_sgep_idx = asc_atomic_add(ub_layout->rank_histogram + dst_rank_idx, 1);
                    if (dst_rank_idx != rank_idx) {
                        ub_layout->sgep_idx[entry_idx] = dst_sgep_idx;
                    } else {
                        ub_layout->local_copy_dst_slot_idx[i] = dst_sgep_idx;
                    }
                }
                if (group_lane_idx < kNumTopk)
                    ub_layout->dst_rank_idx[entry_idx] = dst_rank_idx;
            }
        }
        asc_syncthreads();

        // Create logical SQE range
        if (thread_idx < kNumRanks) {
            // TODO: add `vec_core_idx` into shuffling
            const auto peer_rank_idx = (rank_idx + thread_idx) % kNumRanks;
            if (peer_rank_idx != rank_idx) {
                // 每 peer 保留 ceil(unique_tokens/6) 个逻辑 SQE，一条 SQE 的所有 pair 都发给同一 peer。
                // 不同 peer 的尾 SQE 不能合并，否则不能共用远端 header/address。
                ub_layout->lsqe_base_idx[peer_rank_idx] = asc_atomic_add(
                    &ub_layout->lsqe_counter,
                    math::ceil_div(ub_layout->rank_histogram[peer_rank_idx], kNumSGEPairsPerSQE));
            }
        }

        // Reduce all vector cores' expert histogram results.
        if (thread_idx < kNumExperts) {
            asc_atomic_add(
                workspace_layout.get_local_expert_histogram_ptr(thread_idx),
                // 高位是一份 AIV arrival，低位为专家 count；空闲 AIV 也贡献 arrival，确保 receiver 能等待固定核数。
                reduction::kReductionIdentity | ub_layout->expert_histogram[thread_idx]);
        }

        EP_STATIC_ASSERT(kNumVecCores <= 2 * warpSize and kNumVecCores <= kNumMaxVecCores);
        if (thread_idx < kNumRanks) {
            // local-rank-histogram[kNumVecCores, kNumRanks]
            auto ptr = workspace_layout.get_local_rank_histogram_ptr(vec_core_idx, thread_idx);
            *ptr = reduction::kReductionIdentity | ub_layout->rank_histogram[thread_idx];
            asc_threadfence();
        }
        asc_syncthreads();
        // 本地 rank histogram 先以 arrival=1 表示原始 count；计算核读完上下两半后写 arrival=2 的 exclusive prefix。
        // 每个 AIV 等自己行的 arrival=2 才取出 prefix，并清零该暂存行；避免把未算完的 count 当 offset。
        // calculate prefix sum of rank counts
        // local-rank-histogram-new[i, j] = \sum_{k=0}^{i-1} local-rank-histogram[k, j]
        for (int peer_idx = vec_core_idx + warp_idx * kNumVecCores; peer_idx < kNumRanks; peer_idx += kNumVecCores * kNumWarps) {
            const auto lower_ptr = workspace_layout.get_local_rank_histogram_ptr(lane_idx, peer_idx);
            const auto upper_ptr = workspace_layout.get_local_rank_histogram_ptr(lane_idx + warpSize, peer_idx);
            int lower = 0, upper = 0;
            if (lane_idx < kNumVecCores)
                lower = reduction::get_reduction_value(reduction::simt::wait_ready(lower_ptr, 1));
            const auto lower_sum = simt::warp_inclusive_sum(lower, lane_idx);
            const auto lower_total = simt::exchange(lower_sum, warpSize - 1);
            if (lane_idx + warpSize < kNumVecCores)
                upper = reduction::get_reduction_value(reduction::simt::wait_ready(upper_ptr, 1));
            const auto upper_sum = simt::warp_inclusive_sum(upper, lane_idx);
            const auto upper_total = simt::exchange(upper_sum, warpSize - 1);
            // Finish reading both halves before replacing counts with prefixes.
            asc_threadfence();
            if (lane_idx == 0)
                // rank_psum 是本 rank 所有 AIV 发往 peer 的 token 总数；rank_histogram 随后改为当前 AIV 的前缀。
                ub_layout->rank_psum[peer_idx] = lower_total + upper_total;
            if (lane_idx < kNumVecCores)
                *lower_ptr = 2 * reduction::kReductionIdentity | (lower_sum - lower);
            if (lane_idx + warpSize < kNumVecCores)
                *upper_ptr = 2 * reduction::kReductionIdentity | (lower_total + upper_sum - upper);
        }
        // obtain the final psum of rank counts
        // rank-hist[rank_idx] = local-rank-histogram-new[vec_core_idx, rank_idx]
        if (thread_idx < kNumRanks) {
            const auto ptr = workspace_layout.get_local_rank_histogram_ptr(vec_core_idx, thread_idx);
            ub_layout->rank_histogram[thread_idx] = reduction::get_reduction_value(reduction::simt::wait_ready(ptr, 2));
            *ptr = 0;
        }
        asc_syncthreads();

        // Assign slots and create local copies' metadata
        // Also, create the global SGE to entry mapping
        // NOTES: this costs 1 us
        for (int i = group_idx; i < num_iterations; i += num_groups_per_wave) {
            const auto token_idx = vec_core_idx * kNumMaxTokensPerVecCore + i;
            const auto entry_idx = i * kNumTopk + group_lane_idx;
            int dst_rank_idx = -1, dst_slot_idx = -1, global_sgep_idx = -1;
            if (i < kNumMaxTokensPerVecCore and token_idx < num_tokens and group_lane_idx < kNumTopk) {
                // Read rank index
                dst_rank_idx = ub_layout->dst_rank_idx[entry_idx];

                // Assign slots
                if (dst_rank_idx >= 0) {
                    if (dst_rank_idx != rank_idx) {
                        const auto sgep_idx = ub_layout->sgep_idx[entry_idx];
                        // 最终 peer slot = 本 AIV 之前的 token 总数 + 本 AIV 的 token ordinal。
                        // 所有 AIV 的区间不相交；远端 buffer 按 source rank 固定 Tmax 大小，slot 只在本 source 分区内偏移。
                        dst_slot_idx = ub_layout->rank_histogram[dst_rank_idx] + sgep_idx;
                        const auto lsqe_base_idx = ub_layout->lsqe_base_idx[dst_rank_idx];
                        // SGE pair index = 当前 peer 的 logical SQE 基址*6 + peer 内序号；末 SQE 不满 6 时余位保持 -1。
                        global_sgep_idx = lsqe_base_idx * kNumSGEPairsPerSQE + sgep_idx;
                        ub_layout->sgep_to_entry[global_sgep_idx] = entry_idx;
                    } else {
                        dst_slot_idx = ub_layout->rank_histogram[dst_rank_idx] + ub_layout->local_copy_dst_slot_idx[i];
                    }
                }
                ub_layout->dst_slot_idx[entry_idx] = dst_slot_idx;
                // 把去重后的代表 lane 的 slot/SGE 索引保存到 handle；被去重 lane 保持 -1，cached 重放时无需再次 ballot。
                dst_buffer_slot_idx[token_idx * kNumTopk + group_lane_idx] = dst_slot_idx;
                dst_gsge_idx[token_idx * kNumTopk + group_lane_idx] = global_sgep_idx;
            }

            // For local copies
            if (i < kNumMaxTokensPerVecCore and token_idx < num_tokens) {
                if ((asc_ballot(dst_rank_idx == rank_idx) & routing_group_mask) == 0) {
                    if (group_lane_idx == 0)
                        ub_layout->local_copy_dst_slot_idx[i] = -1;
                } else if (dst_rank_idx == rank_idx) {
                    ub_layout->local_copy_dst_slot_idx[i] = dst_slot_idx;
                }
            }
        }
    }
    asc_syncthreads();
    if (thread_idx == 0)
        // 此时本地 slot 与逆映射已写完，Scalar 可开始本地 hidden 拷贝；remote WQE 和 count reduction 可并行继续。
        ub_layout->local_copy_ready = 1;

    // Load descriptors while the scalar pipeline copies local payloads.
    if (thread_idx < kNumRanks and thread_idx != rank_idx)
        handle::HcommPeerInfo::load_peer_info(ub_layout->peer_info + thread_idx, jetty_ptrs,
                              thread_idx < rank_idx ? thread_idx : thread_idx - 1);
    if (warp_idx == 1 and kNumRanks > 1) {
        handle::HcommJettyInfo::load_jetty_info(&ub_layout->jetty_info, jetty_ptrs, vec_core_idx, lane_idx);
        // Check SQ capacity before releasing other warps to write WQEBBs.
        if (lane_idx == 0 and static_cast<uint64_t>(ub_layout->lsqe_counter) * kNumWQEBBsPerSQESlot > handle::kSQDepth)
            __asc_simt_vf::__trap();
    }
    asc_syncthreads();
    if (thread_idx == 0 and kNumRanks > 1 and
        static_cast<uint32_t>(ub_layout->jetty_info.packed_head) % kSQEBBAlignment != 0)
        __asc_simt_vf::__trap();

    if constexpr (not kCachedMode) {
        // scatter rank counts to remote ranks
        for (int peer_idx = vec_core_idx + thread_idx * kNumVecCores; peer_idx < kNumRanks; peer_idx += kNumVecCores * kNumThreads) {
            auto ptr = layout::get_sym_ptr(
                workspace_layout.get_common_signals(),
                workspace_layout.get_remote_rank_histogram_ptr(rank_idx),
                peer_idx
            );
            // 发送的是每 source->peer 的 rank 去重 token 总数，不是专家展开行数。
            // 这条 peer 的总数由负责它的 AIV 写一次，status 高位编码完整 kNumVecCores 到达。
            *ptr = reduction::make_reduction_status(kNumVecCores, static_cast<int>(ub_layout->rank_psum[peer_idx]));
        }
    }

    // Write SQEs
    static constexpr int kNumWQEBBLanes = handle::kNumWQEBBBytes / sizeof(uint64_t);
    static constexpr int kNumWriteHeaderLanes = sizeof(AscendC::HcommUrmaSqeCtx) / sizeof(uint64_t);
    static constexpr int kNumSGEPairLanes = 2 * sizeof(AscendC::HcommUrmaSgeCtx) / sizeof(uint64_t);
    const int num_sqes = ub_layout->lsqe_counter;

    const int num_sqes_per_warp = math::ceil_div(num_sqes, kNumWarps);
    const int start_lsqe_idx = warp_idx * num_sqes_per_warp;
    const int end_lsqe_idx = math::min(start_lsqe_idx + num_sqes_per_warp, num_sqes);
    for (int i = start_lsqe_idx; i < end_lsqe_idx; ++ i) {
        // Allocate physical SQE
        int psqe_idx = 0;
        if (lane_idx == 0)
            psqe_idx = asc_atomic_add(&ub_layout->psqe_counter, 1);
        psqe_idx = simt::exchange(psqe_idx, 0);

        // Now we need to know the SGEs' source and destination (rank and slot)
        // Also, we use `entry_idx` to see whether the SGE is valid
        EP_STATIC_ASSERT(kNumSGEPairsPerSQE <= warpSize);
        int entry_idx = -1, dst_rank_idx = -1, dst_slot_idx = -1;
        if (lane_idx < kNumSGEPairsPerSQE) {
            const auto global_sge_idx = psqe_idx * kNumSGEPairsPerSQE + lane_idx;
            entry_idx = ub_layout->sgep_to_entry[global_sge_idx];
            if (entry_idx >= 0) {
                dst_rank_idx = ub_layout->dst_rank_idx[entry_idx];
                dst_slot_idx = ub_layout->dst_slot_idx[entry_idx];
            }
        }

        // Create SQE
        // 48B header + 16B * 12 SGEs <= 4 x 64B WQEBBs
        // Header uses 6 lanes
        // SGE pair uses 4 lanes
        // For example
        // Lane 6/7: SGE 0, SGEP 0 token data (the token index is from lane 0)
        //   - ptr = math::advance_ptr(x, token_idx * kHiddenBytes)
        // Lane 8/9: SGE 1, SGEP 0 token meta (the token index is from lane 0)
        //   - ptr = math::advance_ptr(metadata_send_buffer, token_idx * token_layout.get_num_bytes(true))
        // Lane 10/11: SGE 2, SGEP 1 token data (the token index is from lane 1)
        //   - ptr = math::advance_ptr(x, token_idx * kHiddenBytes)
        // Lane 12/13: SGE 3, SGEP 1 token meta (the token index is from lane 1)
        //   - ptr = math::advance_ptr(metadata_send_buffer, token_idx * token_layout.get_num_bytes(true))
        // ...
        EP_STATIC_ASSERT(kNumSGEPairsPerSQE == 6);
        const auto num_sgeps = __reduce_add(entry_idx >= 0);
        // 每个 token 用两份 SGE：原始 x hidden（不先打包复制）+ GM metadata（SF、top-k/weight、source ID）。
        // 接收地址连续写 [hidden|metadata]，因此一 SQE 可串联最多 6 个同 peer token。
        const auto num_sges = num_sgeps * 2;
        const auto num_wqebbs = math::ceil_div(
            static_cast<int>(sizeof(AscendC::HcommUrmaSqeCtx) + num_sges * sizeof(AscendC::HcommUrmaSgeCtx)),
            handle::kNumWQEBBBytes);
        const auto sqe_dst_rank_idx = simt::exchange(dst_rank_idx, 0);
        const auto sqe_dst_slot_idx = simt::exchange(dst_slot_idx, 0);
        const auto peer_info = ub_layout->peer_info + sqe_dst_rank_idx;
        const auto peer_info_words = reinterpret_cast<__ubuf__ uint64_t*>(peer_info);
        const auto sgep_idx = lane_idx < kNumWriteHeaderLanes ? 0 :
            math::min((lane_idx - kNumWriteHeaderLanes) / kNumSGEPairLanes, kNumSGEPairsPerSQE - 1);
        const auto sgep_entry_idx = simt::exchange(entry_idx, sgep_idx);

        const auto slot_head =
            static_cast<uint32_t>(ub_layout->jetty_info.packed_head) +
            psqe_idx * kNumWQEBBsPerSQESlot;
        const auto wqebb_head = slot_head + lane_idx / kNumWQEBBLanes;
        // SQ 环通过 masked index 定位 WQEBB，owner bit 随环回绕翻转，区分旧队列内容与本代新 WQE。
        const auto owner_bit = static_cast<uint64_t>((wqebb_head & handle::kSQDepth) == 0) << 31;
        const auto sq_wqebb_idx = wqebb_head % handle::kSQDepth;

        uint64_t sqe_word = 0;
        if (lane_idx == 0) {
            sqe_word = handle::kWriteSQEHeaderTemplate | owner_bit | static_cast<uint64_t>(sq_wqebb_idx);
            // 最后一条物理 SQE 请求强序和 CQE；后续 drain 通过该完成点回收队列，不能只看 doorbell 已发。
            // Request strong order and CQE for the last physical SQE.
            if (psqe_idx + 1 == num_sqes)
                sqe_word |= handle::kWriteSQEFinalFlags;
        } else if (lane_idx < kNumWriteHeaderLanes) {
            sqe_word = peer_info_words[lane_idx - 1];
            if (lane_idx == 1)
                sqe_word |= static_cast<uint64_t>(num_sges) << 24;  // num_sges
            if (lane_idx == 5)
                // remote peer base 加“当前 source rank 分区”的 recv_buffer_offset，再加去重后的 peer slot*token stride。
                // 相同 token 发给不同 rank 可以共享本地 x/metadata 源地址，各目标拥有独立接收地址。
                sqe_word += recv_buffer_offset + static_cast<uint64_t>(sqe_dst_slot_idx) * num_token_bytes;  // remote_addr
        } else if (lane_idx < kNumWriteHeaderLanes + num_sgeps * kNumSGEPairLanes) {
            const auto sgep_lane_idx = (lane_idx - kNumWriteHeaderLanes) % kNumSGEPairLanes;
            const auto token_idx = vec_core_idx * kNumMaxTokensPerVecCore + sgep_entry_idx / kNumTopk;
            if (sgep_lane_idx == 0) {
                sqe_word = static_cast<uint64_t>(num_hidden_bytes);
            } else if (sgep_lane_idx == 1) {
                sqe_word = reinterpret_cast<uint64_t>(x) + static_cast<uint64_t>(token_idx) * num_hidden_bytes;
            } else if (sgep_lane_idx == 2) {
                sqe_word = static_cast<uint64_t>(num_metadata_bytes);
            } else {
                sqe_word = reinterpret_cast<uint64_t>(metadata_send_buffer) +
                    static_cast<uint64_t>(token_idx) * num_metadata_bytes;
            }
        // 每逻辑 SQE 固定预留 4 个 WQEBB；实际条目不足时以 NOP 补齐，使 slot 步长和 SQ 对齐不变。
        } else if (lane_idx >= num_wqebbs * kNumWQEBBLanes and lane_idx % kNumWQEBBLanes == 0) {
            sqe_word = handle::kNopSQEHeaderTemplate | owner_bit | static_cast<uint64_t>(sq_wqebb_idx);
        }

        // TODO: we can assume SQ depth is power of 2
        // TODO: some of the expression can use i32
        asc_stcg(
            reinterpret_cast<__gm__ uint64_t*>(ub_layout->jetty_info.sq_base_addr) +
                static_cast<uint64_t>(sq_wqebb_idx) * kNumWQEBBLanes + lane_idx % kNumWQEBBLanes,
            sqe_word);

        // TODO(Kexing): `syncthreads` and notify doorbell
        // NOTES: if you do `syncthreads`, please make sure the `i` iteration count is same
    }

    // WQE 可以先写，但 source metadata 尚在 Scalar/MTE3 生产时不能让网络读取。
    // SIMT 等 metadata_ready 后复制本地 metadata；Scalar 随后等整个 VF 完成才敲 remote doorbell。
    while (ub_layout->metadata_ready != 1)
        simt::nop<15>();
    asc_syncthreads();
    const auto num_iters = math::max(0, math::min(kNumMaxTokensPerVecCore, num_tokens - vec_core_idx * kNumMaxTokensPerVecCore));
    for (int i = warp_idx; i < num_iters; i += kNumWarps) {
        const auto dst_slot_idx = ub_layout->local_copy_dst_slot_idx[i];
        if (dst_slot_idx >= 0) {
            const auto token_idx = vec_core_idx * kNumMaxTokensPerVecCore + i;
            const auto src = reinterpret_cast<__gm__ const uint64_t*>(metadata_send_buffer + static_cast<int64_t>(token_idx) * num_metadata_bytes);
            const auto dst = reinterpret_cast<__gm__ uint64_t*>(
                reinterpret_cast<uint64_t>(&workspace_layout.get_common_signals()) + recv_buffer_offset +
                static_cast<uint64_t>(dst_slot_idx) * num_token_bytes + num_hidden_bytes);
            for (int word_idx = lane_idx; word_idx < num_metadata_bytes / sizeof(uint64_t); word_idx += warpSize)
                asc_stcg(dst + word_idx, src[word_idx]);
        }
    }

    // cached 模式复用原 rank/expert prefix，所以本地 metadata 拷贝后无需重做两级计数汇总。
    if constexpr (kCachedMode)
        return;

    // AIV0 负责专家计数：先等本 rank 所有 AIV histogram，再给专家 owner rank 原子贡献一份 source count。
    // 接收端等所有 rank（包括 count=0）到齐；只参与 count reduction，不证明 URMA 数据到达。
    // Expert reduction: wait for all vector cores, send to remote and clean reduction workspace
    if (vec_core_idx == 0 and thread_idx < kNumExperts) {
        const auto status = reduction::simt::wait_ready(
            workspace_layout.get_local_expert_histogram_ptr(thread_idx), kNumVecCores);

        // Send to the responsible rank
        const auto dst_rank_idx = thread_idx / kNumExpertsPerRank;
        const auto num_local_tokens = reduction::get_reduction_value(status);
        const auto dst_expert_idx = thread_idx - dst_rank_idx * kNumExpertsPerRank;
        asc_atomic_add(
            layout::get_sym_ptr(workspace_layout.get_common_signals(),
                                workspace_layout.get_remote_expert_histogram_ptr(dst_expert_idx), dst_rank_idx),
            reduction::kReductionIdentity | num_local_tokens
        );

        // Clean for later usages
        *workspace_layout.get_local_expert_histogram_ptr(thread_idx) = 0;
    }

    // Expert reduction: wait for all ranks' arrival
    if (vec_core_idx == 0 and thread_idx < kNumExpertsPerRank) {
        const auto status = reduction::simt::wait_ready<kNumTimeoutCycles>(
            workspace_layout.get_remote_expert_histogram_ptr(thread_idx), kNumRanks);

        // Record the token count into UB, GM and host buffers
        // TODO: the results to host may be not aligned
        const auto num_recv_tokens = reduction::get_reduction_value(status);
        if constexpr (kDoCPUSync) {
            *host_workspace_layout.get_host_expert_count_ptr(thread_idx) =
                reduction::make_reduction_status(kNumRanks, math::align(num_recv_tokens, kExpertAlignment));
        }
        // 真实专家行数用于统计；host 收到的 aligned count 用于分配/padding，不可将对齐行计入实际专家负载。
        num_unaligned_recv_tokens_per_expert[thread_idx] = num_recv_tokens;
        ub_layout->expert_histogram[thread_idx] = num_recv_tokens;
        if (cumulative_local_expert_recv_stats != nullptr)
            asc_atomic_add(cumulative_local_expert_recv_stats + thread_idx, num_recv_tokens);

        // Clean for later usages
        *workspace_layout.get_remote_expert_histogram_ptr(thread_idx) = 0;
    }

    // Rank reduction: wait for all vector cores, send to remote, wait for all ranks and clean reduction workspace
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads);
    EP_STATIC_ASSERT(kNumVecCores > 1);
    if (vec_core_idx == 1 and thread_idx < kNumRanks) {
        // AIV1 收集每个 source rank 发往本 rank 的唯一 token 数，生成 rank-major 接收流 prefix。
        // Wait remote arrival
        ub_layout->rank_psum[thread_idx] = reduction::get_reduction_value(
            reduction::simt::wait_ready<kNumTimeoutCycles>(
                workspace_layout.get_remote_rank_histogram_ptr(thread_idx), kNumVecCores));

        // Clean workspace for next usages
        *workspace_layout.get_remote_rank_histogram_ptr(thread_idx) = 0;
    }
    asc_syncthreads();

    // Create expert psum layout
    if (vec_core_idx == 0 and thread_idx < warpSize) {
        int inclusive_psum = 0;
        #pragma unroll
        for (int i = 0; i < math::ceil_div(kNumExpertsPerRank, warpSize); ++ i) {
            const auto idx = i * warpSize + thread_idx;
            const auto value = idx < kNumExpertsPerRank ? ub_layout->expert_histogram[idx] : 0;
            const auto aligned = math::align(value, kExpertAlignment);
            const auto inclusive_warp_psum = simt::warp_inclusive_sum(aligned, thread_idx);
            const auto exclusive_warp_psum = inclusive_warp_psum - aligned;

            // Store into global memory
            // NOTES: elements in psum are aligned, but the current value is not aligned
            if (idx < kNumExpertsPerRank)
                // 专家 end = 前序专家 aligned 容量之和 + 当前真实 count；下一专家 start=align(当前 end)。
                // 因此 prefix 同时保留有效行尾部和专家间 padding，不是单纯的累计有效 route 数。
                psum_num_recv_tokens_per_expert[idx] = inclusive_psum + exclusive_warp_psum + value;

            // Maintain in-loop psum
            inclusive_psum += simt::exchange(inclusive_warp_psum, warpSize - 1);
        }
    }

    // Create rank psum layout
    if (vec_core_idx == 1 and thread_idx < warpSize) {
        int inclusive_psum = 0;
        #pragma unroll
        for (int i = 0; i < math::ceil_div(kNumRanks, warpSize); ++ i) {
            const auto idx = i * warpSize + lane_idx;
            const auto value = idx < kNumRanks ? ub_layout->rank_psum[idx] : 0;
            const auto inclusive_warp_psum = simt::warp_inclusive_sum(value, lane_idx);
            const auto psum = inclusive_psum + inclusive_warp_psum;

            // Write global memory
            if (idx < kNumRanks) {
                psum_num_recv_tokens_per_rank[idx] = psum;
                if constexpr (kDoCPUSync) {
                    *host_workspace_layout.get_host_rank_count_ptr(idx) =
                        reduction::make_reduction_status(kNumRanks, value);
                }
            }

            // Maintain in-loop psum
            inclusive_psum += simt::exchange(inclusive_warp_psum, warpSize - 1);
        }
    }
}

template <int kNumRanks, int kNumExperts, int kNumTopk,
          int kExpertAlignment, bool kCachedMode, bool kDoCPUSync,
          int kNumVecCores,
          int kNumHiddenBytes, int kNumSFPacks, int kNumMaxTokensPerRank, int kNumUbBytes,
          int64_t kNumTimeoutCycles, bool kDoBarrier = false,
          int kNumMaxTokensPerVecCore = math::ceil_div(kNumMaxTokensPerRank, kNumVecCores),
          int kNumJetties = kNumVecCores,
          int kNumSGEPairsPerSQE = 6>
__global__ __vector__ void dispatch_impl(
    __gm__ void* x, __gm__ sf_pack_t* sf,
    __gm__ int64_t* topk_idx, __gm__ float* topk_weights,
    __gm__ uint8_t* metadata_send_buffer,
    __gm__ int* cumulative_local_expert_recv_stats,
    __gm__ int* psum_num_recv_tokens_per_rank,
    __gm__ int* psum_num_recv_tokens_per_expert, __gm__ int* num_unaligned_recv_tokens_per_expert,
    __gm__ int* dst_buffer_slot_idx, __gm__ int* dst_gsge_idx,
    __gm__ void* buffer, __gm__ void* workspace, __gm__ void* host_workspace,
    __gm__ void* jetty_ptrs,
    const int rank_idx,
    const int num_tokens,
    const int sf_token_stride,
    const int sf_hidden_stride) {
    // TODO(HUAWEI): kernel launch & init SoC cost 1.9 us, which is slow
    AscendC::InitSocState();

    // Checks
    // NOTES: the one-to-many-rank Jetty is private for this vector core
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRanks);
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0);
    EP_STATIC_ASSERT(math::is_power_of_2(kExpertAlignment));
    EP_STATIC_ASSERT(kNumVecCores > 0);
    EP_STATIC_ASSERT(kNumTopk > 0);
    EP_STATIC_ASSERT(kNumMaxTokensPerRank > 0);
    EP_STATIC_ASSERT(kNumMaxTokensPerVecCore > 0);
    EP_STATIC_ASSERT(kNumSGEPairsPerSQE <= handle::kNumMaxSGEPairsPerSQE);
    EP_STATIC_ASSERT(kNumRanks <= reduction::kReductionMaxHighCount and kNumVecCores <= reduction::kReductionMaxHighCount,
                     "Reduction arrival count exceeds 8 bits");
    EP_STATIC_ASSERT(static_cast<int64_t>(kNumMaxTokensPerRank) * kNumRanks <= reduction::kReductionMaxLowCount,
                     "Per-expert token count exceeds the 24-bit reduction value");
    EP_STATIC_ASSERT(not kDoCPUSync or math::align<int64_t>(
                         static_cast<int64_t>(kNumMaxTokensPerRank) * kNumRanks, kExpertAlignment) <= reduction::kReductionMaxLowCount,
                     "Aligned expert count exceeds the 24-bit host notification value");
    EP_STATIC_ASSERT(math::align<int64_t>(
                         static_cast<int64_t>(kNumMaxTokensPerRank) * kNumRanks * math::min(kNumTopk, kNumExperts / kNumRanks) +
                         static_cast<int64_t>(kExpertAlignment - 1) * (kNumExperts / kNumRanks), kExpertAlignment) <= INT_MAX,
                     "Expanded token count including expert padding overflows int");

    // Each remote peer can leave one partially filled SQE; all SQEs are written before the doorbell.
    static constexpr int64_t kNumMaxRemoteEntries =
        static_cast<int64_t>(kNumMaxTokensPerVecCore) * math::min(kNumTopk, kNumRanks - 1);
    static constexpr int64_t kNumMaxSQEs = math::min(kNumMaxRemoteEntries, math::min<int64_t>(
        static_cast<int64_t>(kNumRanks - 1) * math::ceil_div(kNumMaxTokensPerVecCore, kNumSGEPairsPerSQE),
        (kNumMaxRemoteEntries + static_cast<int64_t>(kNumRanks - 1) * (kNumSGEPairsPerSQE - 1)) / kNumSGEPairsPerSQE));
    EP_STATIC_ASSERT(kNumMaxSQEs * kNumWQEBBsPerSQESlot <= static_cast<int64_t>(UINT16_MAX) + 1,
                     "Dispatch WQEBBs exceed the 16-bit SQ index range");
    // TODO: tune the best Jetty count
    EP_STATIC_ASSERT(kNumJetties == kNumVecCores and kNumJetties <= 64);
    EP_DEVICE_ASSERT(0 <= num_tokens and num_tokens <= kNumMaxTokensPerRank);
    if constexpr (kNumSFPacks > 0)
        EP_DEVICE_ASSERT(sf_token_stride > 0 and sf_hidden_stride > 0);

    const auto vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace);
    const auto host_workspace_layout = layout::EPWorkspaceLayout(host_workspace);

    // hidden 是调用方数据的原始字节；FP8 输入时附 SF，BF16 输入通常无 SF，此 kernel 不执行量化。
    // 非 cached token 带完整 top-k ID/weight/source ID；cached 保留 SF/weight，路由身份来自旧 handle。
    // Token and buffer layouts
    // Cached dispatch reuses source metadata from the handle; only SF and weights travel again.
    const auto token_layout = layout::TokenLayout(
        kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, not kCachedMode);
    const auto recv_buffer = layout::BufferLayout(
        token_layout, kNumRanks, kNumMaxTokensPerRank,
        math::advance_ptr<void>(buffer, 0)
    ).get_rank_buffer(rank_idx);

    // Assign unified buffer
    using ub_layout_t = DispatchUBLayout<kNumExperts, kNumRanks, kNumMaxTokensPerVecCore * kNumTopk, kNumMaxTokensPerVecCore>;
    using ub_layout_with_buffer_t = UBLayoutWithBuffer<ub_layout_t, kNumHiddenBytes, kNumSFPacks,
        kNumTopk, kNumMaxTokensPerVecCore, kNumUbBytes, kCachedMode>;
    EP_STATIC_ASSERT(sizeof(ub_layout_with_buffer_t) <= kNumUbBytes, "Dispatch buffers exceed UB capacity");
    constexpr auto kNumMetadataMTEStages = ub_layout_with_buffer_t::kNumMetadataMTEStages;
    constexpr auto kNumHiddenMTEStages = ub_layout_with_buffer_t::kNumHiddenMTEStages;
    constexpr auto kNumTokensPerMetadataMTE = ub_layout_with_buffer_t::kNumTokensPerMetadataMTE;
    const auto ub = reinterpret_cast<__ubuf__ ub_layout_with_buffer_t*>(0);
    const auto ub_layout = &ub->workspace;
    const auto metadata_token_layout = layout::TokenLayout(
        0, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, not kCachedMode);
    const auto ub_metadata_buffer = layout::BufferLayout(
        metadata_token_layout, kNumMetadataMTEStages, kNumTokensPerMetadataMTE,
        math::advance_ptr<void>(ub->metadata, 0)
    );

    // 进入时跨 rank 对齐本代生命周期；出发前的 barrier 无法保证本代后续 URMA write 完成。
    // 接收 epilogue 才用带 jetty drain 的 barrier 建立数据可消费边界。
    // Barrier before dispatch
    // NOTES: this costs 4 us (8 ranks)
    // NOTES: marking is for aligning different ranks' time
#if defined(ASCENDC_TRACE_ON)
    static constexpr uint16_t kTraceBarrierBefore = 0x101;
    static constexpr uint16_t kTraceBarrierAfter = 0x102;
    if (vec_core_idx == 0)
        asc_mark_stamp<PIPE_S>(kTraceBarrierBefore);
#endif
    comm::scalar::barrier<kNumRanks, kNumTimeoutCycles, false>(workspace_layout.get_common_signals(), vec_core_idx);
#if defined(ASCENDC_TRACE_ON)
    if (vec_core_idx == 0)
        asc_mark_stamp<PIPE_S>(kTraceBarrierAfter);
#endif

    // Do notify, SQE writes and local copy pipeline using a SIMT persistent worker
    handle::HcommJetty jetty(&ub_layout->jetty_info, nullptr, nullptr, vec_core_idx);

    {
        static constexpr int kNumThreads = 512;
        ub_layout->local_copy_ready = 0;
        ub_layout->metadata_ready = 0;

        // Fence S -> V
        asc_sync_notify(PIPE_S, PIPE_V, static_cast<event_t>(0));
        asc_sync_wait(PIPE_S, PIPE_V, static_cast<event_t>(0));

        // Call VF
        // TODO(HUAWEI): launching this VF costs 0.5 us, which is slow
        AscendC::Simt::VF_CALL<
            simt_persistent_worker<kNumMaxTokensPerRank, kNumRanks, kNumExperts, kNumTopk, kExpertAlignment, kCachedMode, kDoCPUSync,
                                   kNumVecCores, kNumThreads, kNumSGEPairsPerSQE, kNumTimeoutCycles>>(
            AscendC::Simt::Dim3(kNumThreads, 1, 1),
            workspace_layout, host_workspace_layout, ub_layout,
            static_cast<__gm__ const uint8_t*>(x), metadata_send_buffer,
            static_cast<__gm__ const uint64_t*>(jetty_ptrs),
            topk_idx, cumulative_local_expert_recv_stats,
            psum_num_recv_tokens_per_rank, psum_num_recv_tokens_per_expert, num_unaligned_recv_tokens_per_expert,
            dst_buffer_slot_idx, dst_gsge_idx,
            reinterpret_cast<uint64_t>(recv_buffer.base) - reinterpret_cast<uint64_t>(workspace),
            kNumHiddenBytes, static_cast<int>(token_layout.get_num_bytes(true)),
            static_cast<int>(token_layout.get_num_bytes()),
            num_tokens, vec_core_idx, rank_idx
        );
    }

    // metadata 采用多 stage GM->UB->GM 打包：SF 支持跨 token/channel stride，top-k ID 与 weight 另行读入。
    // 统一连续 metadata_send_buffer 供网络 SGE 引用，hidden SGE 则直接引用 x，不需要完整 token 预打包。
    auto preload_metadata = [&](const int token_idx, const int stage_idx) __aicore__ {
        const auto ub_metadata = ub_metadata_buffer.get_token_buffer(stage_idx * kNumTokensPerMetadataMTE);
        const auto num_metadata_bytes = static_cast<uint32_t>(metadata_token_layout.get_num_bytes());
        const auto num_tokens_in_stage = math::min(
            kNumTokensPerMetadataMTE,
            math::max(0, math::min(num_tokens, (vec_core_idx + 1) * kNumMaxTokensPerVecCore) - token_idx));

        // Copy metadata within this AIV's contiguous token range
        // 首次预置 MTE3->MTE2 credit；后续复用必须等待旧 metadata 的 UB->GM 拷贝不再读此槽。
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));
        if constexpr (kNumSFPacks > 0) {
            const auto gm_sf = math::advance_ptr<__gm__ sf_pack_t>(
                sf, static_cast<int64_t>(token_idx) * sf_token_stride * sizeof(sf_pack_t));
            // 连续 SF 直接按 token burst 读取；非连续 channel scale 使用二维 NDDMA，目标统一为紧凑 token record。
            if (sf_hidden_stride == 1) {
                asc_copy_gm2ub_align(
                    ub_metadata.template get_sf_ptr<__ubuf__ sf_pack_t>(), gm_sf,
                    /* n_burst= */ static_cast<uint16_t>(num_tokens_in_stage),
                    /* len_burst= */ kNumSFPacks * sizeof(sf_pack_t),
                    /* left_padding_num= */ 0, /* right_padding_num= */ 0,
                    /* enable_constant_pad= */ false,
                    /* l2_cache_mode= */ static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                    /* src_stride= */ static_cast<uint64_t>(sf_token_stride) * sizeof(sf_pack_t),
                    /* dst_stride= */ num_metadata_bytes
                );
            } else {
                const nddma_desc sf_desc(
                    nddma_desc::loop_desc(
                        static_cast<uint32_t>(num_tokens_in_stage),
                        num_metadata_bytes / sizeof(sf_pack_t),
                        static_cast<uint32_t>(sf_token_stride)),
                    nddma_desc::loop_desc(
                        kNumSFPacks, 1, static_cast<uint32_t>(sf_hidden_stride))
                );
                nddma_out_to_ub(
                    ub_metadata.template get_sf_ptr<__ubuf__ sf_pack_t>(), gm_sf,
                    /* sid= */ 0, sf_desc, /* pad_val= */ 0, CONSTANT_PADDING,
                    /* l2_cache_ctl= */ static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV)
                );
            }
        }
        if constexpr (not kCachedMode) {
            asc_copy_gm2ub_align(
                ub_metadata.template get_topk_idx_ptr<__ubuf__ uint8_t>(),
                math::advance_ptr<__gm__ uint8_t>(topk_idx, static_cast<int64_t>(token_idx) * kNumTopk * sizeof(int64_t)),
                /* n_burst= */ static_cast<uint16_t>(num_tokens_in_stage),
                /* len_burst= */ kNumTopk * sizeof(int64_t),
                /* left_padding_num= */ 0, /* right_padding_num= */ 0,
                /* enable_constant_pad= */ false,
                /* l2_cache_mode= */ static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                /* src_stride= */ static_cast<uint64_t>(kNumTopk) * sizeof(int64_t),
                /* dst_stride= */ num_metadata_bytes
            );
        }
        if (topk_weights != nullptr) {
            asc_copy_gm2ub_align(
                ub_metadata.template get_topk_weights_ptr<__ubuf__ float>(), topk_weights + token_idx * kNumTopk,
                /* n_burst= */ static_cast<uint16_t>(num_tokens_in_stage),
                /* len_burst= */ kNumTopk * sizeof(float),
                /* left_padding_num= */ 0, /* right_padding_num= */ 0,
                /* enable_constant_pad= */ false,
                /* l2_cache_mode= */ static_cast<uint8_t>(LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV),
                /* src_stride= */ static_cast<uint64_t>(kNumTopk) * sizeof(float),
                /* dst_stride= */ num_metadata_bytes
            );
        }
        asc_sync_notify(PIPE_MTE2, PIPE_S, static_cast<event_t>(stage_idx));
    };

    // Start metadata and local copies as soon as SIMT publishes the local slots.
    EP_STATIC_ASSERT(kNumMetadataMTEStages <= 8, "Insufficient metadata events");
    EP_STATIC_ASSERT(kNumHiddenMTEStages > 0 and kNumHiddenMTEStages <= 8, "Insufficient hidden-buffer events");
    EP_STATIC_ASSERT(kNumTokensPerMetadataMTE > 0);
    const auto num_tokens_for_vec_core = math::max(
        0, math::min(kNumMaxTokensPerVecCore, num_tokens - vec_core_idx * kNumMaxTokensPerVecCore));
    const auto num_metadata_batches = math::ceil_div(num_tokens_for_vec_core, kNumTokensPerMetadataMTE);
    const auto num_active_stages = math::min(num_metadata_batches, kNumMetadataMTEStages);

    for (int stage_idx = 0; stage_idx < num_active_stages; ++ stage_idx)
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));
    for (int stage_idx = 0; stage_idx < num_active_stages; ++ stage_idx)
        preload_metadata(vec_core_idx * kNumMaxTokensPerVecCore + stage_idx * kNumTokensPerMetadataMTE, stage_idx);

    // Prepare metadata while SIMT computes the routing slots.
    for (int batch_idx = 0; batch_idx < num_metadata_batches; ++ batch_idx) {
        const auto stage_idx = batch_idx % kNumMetadataMTEStages;
        const auto first_metadata_idx = batch_idx * kNumTokensPerMetadataMTE;
        const auto num_tokens_in_stage = math::min(kNumTokensPerMetadataMTE, num_tokens_for_vec_core - first_metadata_idx);
        asc_sync_wait(PIPE_MTE2, PIPE_S, static_cast<event_t>(stage_idx));
        if constexpr (not kCachedMode) {
            for (int i = 0; i < num_tokens_in_stage; ++ i) {
                const auto token_idx = vec_core_idx * kNumMaxTokensPerVecCore + first_metadata_idx + i;
                const auto ub_metadata = ub_metadata_buffer.get_token_buffer(stage_idx * kNumTokensPerMetadataMTE + i);
                // 全局 source ID=source rank*Tmax+本地 token；Tmax 是注册容量 stride，不随本次 T 改变。
                *ub_metadata.template get_src_token_global_idx_ptr<__ubuf__ int>() = rank_idx * kNumMaxTokensPerRank + token_idx;
            }
        }
        const auto first_token_idx = vec_core_idx * kNumMaxTokensPerVecCore + first_metadata_idx;
        const auto num_metadata_bytes = static_cast<uint32_t>(metadata_token_layout.get_num_bytes());
        asc_sync_notify(PIPE_S, PIPE_MTE3, static_cast<event_t>(stage_idx));
        asc_sync_wait(PIPE_S, PIPE_MTE3, static_cast<event_t>(stage_idx));
        asc_copy_ub2gm_align(
            metadata_send_buffer + static_cast<int64_t>(first_token_idx) * num_metadata_bytes,
            ub_metadata_buffer.get_token_buffer(stage_idx * kNumTokensPerMetadataMTE).template get_base_ptr<__ubuf__ uint8_t>(),
            static_cast<uint16_t>(num_tokens_in_stage), num_metadata_bytes,
            asc_store_l2_cache_mode::NOTALLOC_CLEAN,
            static_cast<uint64_t>(num_metadata_bytes), num_metadata_bytes);
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));
        const auto next_batch_idx = batch_idx + kNumMetadataMTEStages;
        if (next_batch_idx < num_metadata_batches)
            preload_metadata(vec_core_idx * kNumMaxTokensPerVecCore + next_batch_idx * kNumTokensPerMetadataMTE, stage_idx);
    }
    for (int stage_idx = 0; stage_idx < num_active_stages; ++ stage_idx)
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));
    asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);
    // 所有活跃 metadata stage 的 MTE3 都已 drain；此 flag 发布后 SIMT/URMA 可读取 GM metadata。
    ub_layout->metadata_ready = 1;

    // Scalar 等 SIMT 最终 slot 映射；不必等 SIMT 完成远端 WQE/全局 count，从而重叠本地 DMA。
    while (ub_layout->local_copy_ready != 1)
        asm volatile("nop");

    for (int stage_idx = 0; stage_idx < kNumHiddenMTEStages; ++ stage_idx)
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));

    int hidden_stage_idx = 0;
    for (int i = 0; i < num_tokens_for_vec_core; ++ i) {
        const auto dst_slot_idx = ub_layout->local_copy_dst_slot_idx[i];
        if (dst_slot_idx >= 0) {
            const auto token_idx = vec_core_idx * kNumMaxTokensPerVecCore + i;
            const auto ub_hidden = ub->hidden[hidden_stage_idx];

            // Copy token data into UB
            asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(hidden_stage_idx));
            asc_copy_gm2ub_align(
                ub_hidden,
                math::advance_ptr<__gm__ uint8_t>(x, static_cast<int64_t>(token_idx) * kNumHiddenBytes),
                /* n_burst= */ 1, /* len_burst= */ kNumHiddenBytes,
                /* left_padding_num= */ 0, /* right_padding_num= */ 0,
                /* enable_constant_pad= */ false,
                /* l2_cache_mode= */ asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                /* src_gap= */ 0, /* dst_gap= */ 0
            );
            asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(hidden_stage_idx));

            // 本地 token 同样只复制一次，落入本 source rank 的 rank-major 分区；多个本地专家副本由接收 epilogue 生成。
            // Copy token data into the local receive buffer
            asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(hidden_stage_idx));
            asc_copy_ub2gm_align(
                recv_buffer.get_token_buffer(dst_slot_idx).template get_hidden_ptr<__gm__ uint8_t>(),
                ub_hidden,
                /* n_burst= */ 1, /* len_burst= */ kNumHiddenBytes,
                /* l2_cache_mode= */ asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                /* dst_gap= */ 0, /* src_gap= */ 0
            );
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(hidden_stage_idx));
            hidden_stage_idx = (hidden_stage_idx + 1) % kNumHiddenMTEStages;
        }
    }
    for (int stage_idx = 0; stage_idx < kNumHiddenMTEStages; ++ stage_idx)
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(stage_idx));
    asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);

    // metadata、本地 hidden 和 SIMT descriptors 分别完成后才敲门铃；WQE 已存在不等于硬件已开始读取。
    // 此 issue kernel 默认不等待本代全部 URMA 完成，x 与 metadata_send_buffer 必须存活到 epilogue drain。
    // Wait until SIMT finishes routing and count reduction
    // TODO: ring DB overlappingly
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    if constexpr (kNumRanks > 1)
        EP_DEVICE_ASSERT(kNumMaxSQEs * kNumWQEBBsPerSQESlot <= handle::kSQDepth);
    const auto num_sqes = ub_layout->lsqe_counter;
    if (num_sqes > 0) {
        // head 的单位为 WQEBB，不是 token 或 SGE；一个固定 logical SQE slot 占 4 个 WQEBB。
        jetty.advance_sq(num_sqes * kNumWQEBBsPerSQESlot);
        EP_DEVICE_ASSERT(static_cast<uint32_t>(ub_layout->jetty_info.packed_head) % kSQEBBAlignment == 0);
        jetty.ring_doorbell();
    }

    // benchmark only: wait for all urma traffic to measure URMA latency
    if constexpr (kDoBarrier)
        comm::scalar::barrier<
            kNumRanks, kNumTimeoutCycles, true, true, true, kNumVecCores, kNumJetties>(
                workspace_layout.get_common_signals(), vec_core_idx, jetty_ptrs);
}

}  // namespace deep_ep
