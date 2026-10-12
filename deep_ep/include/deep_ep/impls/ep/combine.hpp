#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/comm/handle.hpp>
#include <deep_ep/impls/ep/combine_utils.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/common/bitset.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include <kernel_operator.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>
#include <simt_api/device_functions.h>
#include <simt_api/device_warp_functions.h>

namespace deep_ep {

// Combine 是 rank 去重 dispatch 的逆过程：一个网络接收 token 对应本卡多个专家结果行。
// 先在专家 rank 汇合这些行，再向原 source rank 回传一份 BF16 rank partial；源端 epilogue 汇合各 rank partial。
// 该 kernel 不执行 GMM，也不对 hidden 乘 top-k weight：传入的 x 由调用方决定是否已加权。
// topk_weights 若提供，是待恢复的独立元数据，原样返回，并非在这里用于 hidden 数值乘法。
struct CombineTask {
    int16_t rank_idx;
    int16_t num_tokens;
    // rank_slot_idx 属于 dispatch 的紧凑 rank-major 接收流；同 task 内 token stride 是 AIV 核数，不是 1。
    int32_t rank_slot_idx;
};

template <int kNumRanks, int kNumTopk, int kHidden, int kNumTokensPerTask, int kNumStages, int kNumUbBytes>
struct CombineUBLayout {
    static constexpr int kNumMetadataElems = kNumTopk + 2;
    static constexpr int kNumUbMetadataElems = math::max(warpSize, kNumMetadataElems + 1);
    static constexpr int kNumHiddenBytes = kHidden * sizeof(bfloat16_t);
    static constexpr int kNumTopkWeightBytes = math::align<int>(kNumTopk * sizeof(float), kNumUbAlignmentBytes);
    static constexpr int kNumTopkWeightElems = kNumTopkWeightBytes / sizeof(float);
    // wire record = BF16[H] + 32B 对齐的 FP32[topk]，不传 FP8 scale；weight 缺省时该布局仍预留 weight 区。
    static constexpr int kNumTokenBytes = kNumHiddenBytes + kNumTopkWeightBytes;
    static constexpr int kNumWQEBBWords = handle::kNumWQEBBBytes / sizeof(uint64_t);
    static constexpr int kNumWQEBBsPerToken = math::ceil_div<int>(sizeof(AscendC::HcommUrmaSqeCtx) + 2 * sizeof(AscendC::HcommUrmaSgeCtx), handle::kNumWQEBBBytes);
    static constexpr int kNumWQEBytes = kNumWQEBBsPerToken * handle::kNumWQEBBBytes;
    static constexpr int kNumWQEWords = kNumWQEBytes / sizeof(uint64_t);
    static constexpr int kNumWQEBBsPerTask = kNumTokensPerTask * kNumWQEBBsPerToken;

    handle::HcommJettyInfo jetty_info;
    alignas(kNumUbAlignmentBytes) handle::HcommPeerInfo peer_infos[kNumRanks];
    alignas(kNumUbAlignmentBytes) int remaining_tokens[kNumRanks];
    alignas(kNumUbAlignmentBytes) int rank_next_recv_slot_idx[kNumRanks];
    alignas(kNumUbAlignmentBytes) int metadata[kNumStages][kNumTokensPerTask][kNumUbMetadataElems];

    static constexpr int kNumFixedBytes =
        math::align(static_cast<int>(sizeof(handle::HcommJettyInfo)), kNumUbAlignmentBytes) +
        math::align(kNumRanks * static_cast<int>(sizeof(handle::HcommPeerInfo)), kNumUbAlignmentBytes) +
        2 * math::align(kNumRanks * static_cast<int>(sizeof(int)), kNumUbAlignmentBytes) +
        math::align(kNumStages * kNumTokensPerTask * kNumUbMetadataElems * static_cast<int>(sizeof(int)), kNumUbAlignmentBytes);
    static constexpr int kNumAvailableHiddenBytes = math::max(0, math::align<int, false>(kNumUbBytes, kNumUbAlignmentBytes) - kNumFixedBytes);
    // 先扣 peer info/task metadata，再按 hidden 行字节数分配最多 32 个 UB 槽；必须容纳一个 token 的全部本地专家行。
    static constexpr int kNumHiddenRingSlots = math::min(32, kNumAvailableHiddenBytes / kNumHiddenBytes);
    using HiddenReducer = RingReducer<kHidden, kNumTopk, kNumHiddenRingSlots>;
    alignas(kNumUbAlignmentBytes) typename HiddenReducer::Buffer hidden;
};

template <int kNumRanks, int kNumTopk, int kNumMaxTokensPerRank, int kNumVecCores,
          int kNumTokensPerTask, typename ub_layout_t, bool kHasTopkWeights>
__simt_vf__ __aicore__ __launch_bounds__(kNumTokensPerTask * warpSize) void load_meta_issue_wqe(
        __ubuf__ ub_layout_t* ub, __gm__ const int* src_metadata, __gm__ const bfloat16_t* x, __gm__ uint8_t* send_buffer, __gm__ uint8_t* combine_recv_buffer, __gm__ const uint32_t* topk_weights,
        const uint64_t combine_recv_buffer_offset, const CombineTask task, const uint32_t task_sqebb_idx,
        const int stage_idx, const int rank_idx, const bool issue_wqe) {
    EP_STATIC_ASSERT(kNumTopk > 0 and kNumTopk <= warpSize);

    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto token_idx = thread_idx / warpSize;
    const auto lane_idx = thread_idx % warpSize;
    const auto valid_token = token_idx < task.num_tokens;
    // 按 dispatch 保存的 [recv token][topk+2] 取专家行号/source ID/master slot；AIV stride 与 get_task 配套。
    const auto gm_token_metadata = src_metadata + (task.rank_slot_idx + token_idx * kNumVecCores) * ub_layout_t::kNumMetadataElems;
    int slot_idx = -1, metadata_value = -1;
    // load metadata from GM to register
    #pragma unroll
    for (int metadata_idx = lane_idx; metadata_idx < ub_layout_t::kNumMetadataElems; metadata_idx += warpSize) {
        metadata_value = valid_token ? gm_token_metadata[metadata_idx] : -1;
        if (metadata_idx < kNumTopk)
            slot_idx = metadata_value;
        else
            ub->metadata[stage_idx][token_idx][metadata_idx] = metadata_value;
    }

    // write compacted metadata to ub (remove all -1 value and write only the valid slot_idx)
    // ballot/popcount 将 -1 route 行去掉，仅压紧本 token 在本卡的专家结果行；这里不再次按专家 ID 去重。
    const auto valid_slot_mask = asc_ballot(slot_idx >= 0);
    const auto num_slots = __popc(valid_slot_mask);
    if (slot_idx >= 0) {
        // 本 lane 前方有效位数就是紧凑下标，所有有效行顺序仍按原 top-k 升序，Scalar 可直接遍历。
        const auto packed_idx = __popc(valid_slot_mask & ((1U << lane_idx) - 1));
        ub->metadata[stage_idx][token_idx][packed_idx] = slot_idx;
    }
    if (lane_idx == 0)
        ub->metadata[stage_idx][token_idx][ub_layout_t::kNumUbMetadataElems - 1] = num_slots;

    // 额外两份 metadata 由固定 lane（topk==32 时通过第二轮加载）取出，再 warp broadcast 到 WQE 构造线程。
    const auto src_token_global_idx = asc_shfl(metadata_value, kNumTopk % warpSize);
    const auto master_topk_idx = asc_shfl(metadata_value, (kNumTopk + 1) % warpSize);

    // copy topk weights
    if constexpr (kHasTopkWeights) {
        if (valid_token and lane_idx < ub_layout_t::kNumTopkWeightElems) {
            // 每 rank 只写自己负责 route 的 float 位模式，其他 j 写零；源端将各 rank 的互斥 weight 位用 OR 恢复。
            // 不要将这段 Scalar 权重拷贝误读成 hidden 乘权重。
            const auto weight = lane_idx < kNumTopk and slot_idx >= 0 ? topk_weights[slot_idx] : 0;
            const auto is_local = task.rank_idx == rank_idx;
            // 回传子槽按 min(rank 数,topk) 预留：rank<topk 时按专家 owner rank 定位，反之按 master top-k 槽定位。
            const auto slot_group_idx = kNumRanks < kNumTopk ? rank_idx : master_topk_idx;
            const auto src_token_idx = src_token_global_idx - rank_idx * kNumMaxTokensPerRank;
            const auto local_slot_idx = static_cast<int64_t>(slot_group_idx) * kNumMaxTokensPerRank + src_token_idx;
            const auto remote_slot_idx = static_cast<int64_t>(task.rank_slot_idx) + token_idx * kNumVecCores;
            const auto dst = (is_local ? combine_recv_buffer : send_buffer) +
                (is_local ? local_slot_idx : remote_slot_idx) * ub_layout_t::kNumTokenBytes + ub_layout_t::kNumHiddenBytes;
            asc_stcg(reinterpret_cast<__gm__ uint32_t*>(dst) + lane_idx, weight);
        }
    }

    const auto first_slot_lane_idx = valid_slot_mask == 0 ? 0 : __ffs(static_cast<int>(valid_slot_mask)) - 1;
    const auto first_slot_idx = asc_shfl(slot_idx, first_slot_lane_idx);

    if (issue_wqe and valid_token and lane_idx < ub_layout_t::kNumWQEWords) {
        const auto token_sqebb_idx = task_sqebb_idx + token_idx * ub_layout_t::kNumWQEBBsPerToken;
        const auto gm_wqe = reinterpret_cast<__gm__ uint64_t*>(ub->jetty_info.sq_base_addr +
            (token_sqebb_idx & (handle::kSQDepth - 1)) * handle::kNumWQEBBBytes);
        uint64_t wqe_word = 0;
        if (lane_idx < 10) {
            const auto peer_info_words = reinterpret_cast<__ubuf__ const uint64_t*>(&ub->peer_infos[task.rank_idx]);
            wqe_word = lane_idx == 0 ? handle::kWriteSQEHeaderTemplate : wqe_word;
            wqe_word = lane_idx > 0 and lane_idx < 6 ? peer_info_words[lane_idx - 1] : wqe_word;
            if (lane_idx == 0) {
                // owner bit 随 SQ 环绕翻转；masked head 指向实际 WQEBB，未敲门铃前仅写描述符，不启动 DMA。
                wqe_word |= static_cast<uint64_t>((token_sqebb_idx & handle::kSQDepth) == 0) << 31;
                wqe_word |= static_cast<uint64_t>(token_sqebb_idx & (handle::kSQDepth - 1));
            }
            wqe_word = lane_idx == 1 ? wqe_word | (2ULL << 24) : wqe_word;
            if (lane_idx == 5) {
                const auto src_token_idx = src_token_global_idx - task.rank_idx * kNumMaxTokensPerRank;
                const auto slot_group_idx = kNumRanks < kNumTopk ? rank_idx : master_topk_idx;
                // 目的端 token ID 必须减 task.rank_idx*Tmax（原 source rank），不能减当前专家 rank。
                // remote_offset 指向该源 rank 的 [subslot][token] rank partial 接收槽。
                const auto remote_offset =
                    (static_cast<int64_t>(slot_group_idx) * kNumMaxTokensPerRank + src_token_idx) * ub_layout_t::kNumTokenBytes;
                wqe_word = peer_info_words[4] + combine_recv_buffer_offset + remote_offset;
            }
            wqe_word = lane_idx == 6 ? static_cast<uint64_t>(ub_layout_t::kNumHiddenBytes) : wqe_word;
            if (lane_idx == 7) {
                const auto token_send_buffer_offset =
                    (static_cast<int64_t>(task.rank_slot_idx) + token_idx * kNumVecCores) * ub_layout_t::kNumTokenBytes;
                const auto token_send_buffer_addr = reinterpret_cast<uint64_t>(send_buffer) + token_send_buffer_offset;
                // 只有一份本地专家结果时，hidden SGE 直接引用 x 的专家结果行，绕过求和与 send_buffer 搬运。
                // 多份结果先在 UB 求和、写 GM send_buffer，再由 URMA 读取；本地目标无 WQE，仍需本地拷贝。
                wqe_word = num_slots == 1 ?
                    reinterpret_cast<uint64_t>(x) + static_cast<int64_t>(first_slot_idx) * ub_layout_t::kNumHiddenBytes :
                    token_send_buffer_addr;
            }
            wqe_word = lane_idx == 8 ? static_cast<uint64_t>(ub_layout_t::kNumTopkWeightBytes) : wqe_word;
            if (lane_idx == 9) {
                const auto token_send_buffer_offset =
                    (static_cast<int64_t>(task.rank_slot_idx) + token_idx * kNumVecCores) * ub_layout_t::kNumTokenBytes;
                wqe_word = reinterpret_cast<uint64_t>(send_buffer) + token_send_buffer_offset + ub_layout_t::kNumHiddenBytes;
            }
        }
        asc_stcg(gm_wqe + lane_idx, wqe_word);
    }
}

template <int kNumRanks>
__simt_vf__ __aicore__ __launch_bounds__(math::align(kNumRanks, warpSize))
void load_combine_peers(__ubuf__ handle::HcommPeerInfo* peers, __gm__ void* jetty_ptrs, const int rank_idx) {
    const auto peer_idx = static_cast<int>(threadIdx.x);
    if (peer_idx < kNumRanks and peer_idx != rank_idx)
        handle::HcommPeerInfo::load_peer_info(peers + peer_idx, static_cast<__gm__ const uint64_t*>(jetty_ptrs),
                              peer_idx < rank_idx ? peer_idx : peer_idx - 1);
}

template <int kNumRanks, int kNumTopk, int kHidden, int kNumMaxTokensPerRank, int kNumVecCores,
          int64_t kNumTimeoutCycles, int kNumUbBytes,
          bool kDoBarrier = false, bool kHasTopkWeights = true>
__global__ __vector__ void combine_impl(__gm__ bfloat16_t* x, __gm__ float* topk_weights, __gm__ int* src_metadata,
                                        __gm__ int* psum_num_recv_tokens_per_rank, __gm__ uint8_t* combine_recv_buffer,
                                        __gm__ uint8_t* send_buffer, __gm__ void* workspace, __gm__ void* jetty_ptrs,
                                        const int rank_idx) {
    AscendC::InitSocState();

    // 任务一次最多 8 个 token，metadata 两 stage 预取；每 8 个 remote task 批量敲一次门铃。
    // 这些常量控制提交/预取粒度，与 GEMM 的 M256 块或 MegaMoE 的 ring 大小无关。
    constexpr auto kNumTokensPerTask = 8;
    constexpr auto kNumStages = 2;
    constexpr auto kNumTasksPerDoorbell = 8;

    EP_STATIC_ASSERT(math::is_power_of_2(kNumTasksPerDoorbell));

    using UBLayout = CombineUBLayout<kNumRanks, kNumTopk, kHidden, kNumTokensPerTask, kNumStages, kNumUbBytes>;
    using HiddenReducer = typename UBLayout::HiddenReducer;
    constexpr auto kNumTokenBytes = UBLayout::kNumTokenBytes;
    constexpr auto kNumWQEBBsPerToken = UBLayout::kNumWQEBBsPerToken;
    constexpr auto kNumWQEBBsPerTask = UBLayout::kNumWQEBBsPerTask;

    EP_STATIC_ASSERT(kNumTokensPerTask > 0);
    EP_STATIC_ASSERT(kNumTokensPerTask <= 8);
    EP_STATIC_ASSERT(kNumTopk <= UBLayout::kNumHiddenRingSlots, "Combine UB cannot hold top-k inputs");
    EP_STATIC_ASSERT(kNumVecCores > 0);
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRanks);
    EP_STATIC_ASSERT(kNumMaxTokensPerRank > 0);

    // SQ 总容量按每 peer 尾 task 独立向上取整估算，另留对齐 NOP；单次 kernel 不允许覆盖未消费 WQE。
    // The padded task bound also covers the final alignment NOP WQEs.
    static constexpr int64_t kNumMaxRemoteTasks = static_cast<int64_t>(kNumRanks - 1) *
        math::ceil_div(math::ceil_div(kNumMaxTokensPerRank, kNumVecCores), kNumTokensPerTask);
    EP_STATIC_ASSERT(kNumMaxRemoteTasks * kNumWQEBBsPerTask + kSQEBBAlignment <= static_cast<int64_t>(UINT16_MAX) + 1,
                     "Combine WQEBBs including task padding and final NOP exceed the 16-bit SQ index range");

    const auto topk_weight_u32 = reinterpret_cast<__gm__ uint32_t*>(topk_weights);
    const auto vec_core_idx = static_cast<int>(AscendC::GetBlockIdx());
    const auto combine_recv_buffer_offset = reinterpret_cast<uint64_t>(combine_recv_buffer) - reinterpret_cast<uint64_t>(workspace);
    auto& signals = *static_cast<__gm__ layout::EPSignals*>(workspace);

    EP_STATIC_ASSERT(sizeof(UBLayout) <= kNumUbBytes, "Combine task stages and hidden ring exceed UB capacity");
    __ubuf__ UBLayout* ub = 0;

    comm::scalar::barrier<kNumRanks, kNumTimeoutCycles, false>(signals.common_signals, vec_core_idx);

    handle::HcommJetty jetty(&ub->jetty_info, nullptr, jetty_ptrs, vec_core_idx);
    if constexpr (kNumRanks > 1) {
        AscendC::Simt::VF_CALL<load_combine_peers<kNumRanks>>(
            AscendC::Simt::Dim3(math::align(kNumRanks, warpSize), 1, 1), ub->peer_infos, jetty_ptrs, rank_idx);
        asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    }
    Bitset<kNumRanks> active_ranks = {};

    int num_total_tasks = 0;
    int num_remote_tasks = 0;
    int num_remote_tokens = 0;

    const auto remaining_tokens = ub->remaining_tokens;
    const auto rank_next_recv_slot_idx = ub->rank_next_recv_slot_idx;
    for (int src_rank_idx = 0; src_rank_idx < kNumRanks; ++ src_rank_idx) {
        const auto recv_slot_idx = src_rank_idx == 0 ? 0 : psum_num_recv_tokens_per_rank[src_rank_idx - 1];
        const auto num_tokens = psum_num_recv_tokens_per_rank[src_rank_idx] - recv_slot_idx;
        // apply a swizzle to vec-cores to ensure balanced jetty workload
        // source-rank 与 AIV 索引联合 swizzle，每核从该 peer 分区内不同位置开始，避免所有 jetty 尾部集中同一核。
        const auto start_idx = (vec_core_idx + src_rank_idx * math::ceil_div(kNumVecCores, kNumRanks)) % kNumVecCores;
        const auto num_iters = start_idx < num_tokens ? math::ceil_div(num_tokens - start_idx, kNumVecCores) : 0;
        remaining_tokens[src_rank_idx] = num_iters;
        // 起点为该 source rank 的紧凑 recv prefix 加 swizzle lane；后续任务沿核数 stride 前进。
        rank_next_recv_slot_idx[src_rank_idx] = recv_slot_idx + start_idx;
        if (num_iters > 0)
            active_ranks.set(src_rank_idx);
        const auto num_tasks = math::ceil_div(num_iters, kNumTokensPerTask);
        num_total_tasks += num_tasks;
        if (src_rank_idx != rank_idx) {
            num_remote_tasks += num_tasks;
            num_remote_tokens += num_iters;
        }
    }

    // Stagger the round-robin peer order across ranks and vector cores.
    int16_t next_rank_idx = (rank_idx + vec_core_idx + 1) % kNumRanks;
    // 轮询下一个有剩余 token 的 peer；Bitset 跳过空 peer，拿 task 后减少计数并维护独立下次 slot。
    const auto get_task = [&]() __aicore__ {
        auto task_rank_idx = next_rank_idx;
        if (remaining_tokens[task_rank_idx] == 0)
            task_rank_idx = active_ranks.find(next_rank_idx);
        const auto rank_slot_idx = rank_next_recv_slot_idx[task_rank_idx];
        int16_t num_task_tokens = math::min(kNumTokensPerTask, remaining_tokens[task_rank_idx]);
        // 一个 task 消费至多 8 个、本核间隔 kNumVecCores 的接收 token；不能把 task 当成 GM 连续行。
        rank_next_recv_slot_idx[task_rank_idx] += num_task_tokens * kNumVecCores;
        remaining_tokens[task_rank_idx] -= num_task_tokens;
        if (remaining_tokens[task_rank_idx] == 0)
            active_ranks.clear(task_rank_idx);
        next_rank_idx = task_rank_idx + 1 == kNumRanks ? 0 : task_rank_idx + 1;
        return CombineTask{task_rank_idx, num_task_tokens, rank_slot_idx};
    };

    const auto sq_base_addr = kNumRanks > 1 ? reinterpret_cast<__gm__ uint8_t*>(ub->jetty_info.sq_base_addr) : nullptr;
    const auto sq_head = kNumRanks > 1 ? static_cast<uint32_t>(ub->jetty_info.packed_head) : 0;
    if constexpr (kNumRanks > 1) {
        EP_DEVICE_ASSERT(sq_head % kSQEBBAlignment == 0);
        EP_DEVICE_ASSERT(num_remote_tasks == 0 or
                         static_cast<int64_t>(num_remote_tasks) * kNumWQEBBsPerTask + kSQEBBAlignment <= handle::kSQDepth);
    }
    HiddenReducer reducer(&ub->hidden);
    CombineTask fetched_tasks[kNumStages];
    uint32_t next_remote_sqebb_idx = sq_head;
    uint32_t fetched_sq_ends[kNumStages];
    const auto issue_load_meta_wqe = [&](const uint8_t stage_idx, const int task_idx) __aicore__ {
        if (task_idx >= num_total_tasks)
            return;
        const auto task = get_task();
        // local source 只走本地 reducer/store；remote source 预写 WQE 并递增实际有效 token 数对应的 SQ head。
        const auto issue_wqe = task.rank_idx != rank_idx;
        fetched_tasks[stage_idx] = task;
        const auto task_sqebb_idx = next_remote_sqebb_idx;
        AscendC::Simt::VF_CALL<load_meta_issue_wqe<kNumRanks, kNumTopk, kNumMaxTokensPerRank,
                                                 kNumVecCores, kNumTokensPerTask, UBLayout, kHasTopkWeights>>(
            AscendC::Simt::Dim3(kNumTokensPerTask * warpSize, 1, 1),
            ub, src_metadata, x, send_buffer, combine_recv_buffer, topk_weight_u32,
            combine_recv_buffer_offset, task, task_sqebb_idx, stage_idx, rank_idx, issue_wqe
        );
        asc_sync_notify(PIPE_V, PIPE_S, static_cast<event_t>(stage_idx));
        if (issue_wqe)
            next_remote_sqebb_idx += task.num_tokens * kNumWQEBBsPerToken;
        // 记录该 stage 实际已准备描述符的右开 head；prefetch 可以准备未来 WQE，但 doorbell 不能越过当前 payload 完成点。
        fetched_sq_ends[stage_idx] = next_remote_sqebb_idx;
    };

    #pragma unroll 1
    for (uint8_t stage_idx = 0; stage_idx < kNumStages; ++ stage_idx) {
        issue_load_meta_wqe(stage_idx, stage_idx);
    }

    int num_issued_remote_tasks = 0;
    uint8_t stage_idx = 0;
    for (int task_iter_idx = 0; task_iter_idx < num_total_tasks; ++ task_iter_idx) {
        const auto current_task = fetched_tasks[stage_idx];
        const auto current_sq_end = fetched_sq_ends[stage_idx];
        const auto num_stage_tokens = current_task.num_tokens;
        const auto is_local = current_task.rank_idx == rank_idx;

        // 等待当前 metadata/WQE VF 完成后才能读 UB 行号；peer descriptors、metadata 与 payload 完成分开同步。
        asc_sync_wait(PIPE_V, PIPE_S, static_cast<event_t>(stage_idx));
        for (int token_idx = 0; token_idx < num_stage_tokens; ++ token_idx) {
            const auto meta = ub->metadata[stage_idx][token_idx];
            const auto recv_slot_idx = current_task.rank_slot_idx + token_idx * kNumVecCores;
            const auto src_token_idx = meta[kNumTopk] - rank_idx * kNumMaxTokensPerRank;
            const auto subslot_idx = kNumRanks < kNumTopk ? rank_idx : meta[kNumTopk + 1];
            // 本地目的地址 = 本 rank 的 [subslot][原 token]；远端 partial 先落 send_buffer 的当前 recv token 行。
            const auto combine_slot_idx = subslot_idx * kNumMaxTokensPerRank + src_token_idx;
            const auto output_addr = is_local ?
                reinterpret_cast<__gm__ bfloat16_t*>(
                    combine_recv_buffer + static_cast<int64_t>(combine_slot_idx) * kNumTokenBytes) :
                reinterpret_cast<__gm__ bfloat16_t*>(send_buffer + static_cast<int64_t>(recv_slot_idx) * kNumTokenBytes);
            const auto num_slots = meta[UBLayout::kNumUbMetadataElems - 1];
            const auto first_slot_idx = num_slots > 0 ? meta[0] : -1;
            // 同 token 本卡所有专家结果行加载到 UB ring，再在本地求和；网络只回传一份 rank partial。
            // 这是 combine 的通信聚合，不是远端 AtomicAdd 到最终输出。
            if (num_slots >= 2) {
                reducer.load_first_2_token(x, meta[0], meta[1], asc_load_l2_cache_mode::NOTALLOC_KEEP);
                for (int slot_idx = 2; slot_idx < num_slots; ++ slot_idx)
                    reducer.load_next_token(x + static_cast<int64_t>(meta[slot_idx]) * kHidden,
                                           asc_load_l2_cache_mode::NOTALLOC_KEEP);
            }
            // local+单行必须做本地 copy；remote+单行由 WQE 直接读 x，不生成 send_buffer hidden。
            if (is_local and num_slots == 1) {
                reducer.load_next_token(
                    x + static_cast<int64_t>(first_slot_idx) * kHidden, asc_load_l2_cache_mode::NOTALLOC_KEEP);
                reducer.store_last_token(output_addr);
            } else if (num_slots > 1) {
                reducer.reduce(num_slots);
                reducer.store_last_token(output_addr);
            }
        }

        // 当前 stage 的 metadata 已消费，可安排两任务之后的预取；hidden payload 的槽生命周期由 RingReducer mutex 保护。
        issue_load_meta_wqe(stage_idx, task_iter_idx + kNumStages);

        // 批量 doorbell 前等 MTE3 写好当前 batch 的 send_buffer；提交 head 只到 current_sq_end，
        // 不能用 prefetch 后的 next_remote_sqebb_idx，否则网络会读尚未完成的未来 partial。
        if (not is_local and (++ num_issued_remote_tasks % kNumTasksPerDoorbell == 0)) {
            asc_sync_notify(PIPE_MTE3, PIPE_S, static_cast<event_t>(stage_idx));
            asc_sync_wait(PIPE_MTE3, PIPE_S, static_cast<event_t>(stage_idx));
            jetty.set_sq_head(current_sq_end);
            jetty.ring_doorbell();
        }

        stage_idx = stage_idx + 1 == kNumStages ? 0 : stage_idx + 1;
    }

    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);

    // 最后 drain MTE3 后补强序 NOP/CQE，覆盖包括最终不足 8 task 的所有 payload。
    // issue kernel 返回仍不等于网络完成；接收 epilogue 的 jetty drain 才回收发送源/队列生命周期。
    // Append a strong-order NOP requesting CQE; the barrier only polls completion.
    if (num_remote_tasks > 0) {
        const auto final_sqebb_idx = sq_head + num_remote_tokens * kNumWQEBBsPerToken;
        const auto new_head = math::align(final_sqebb_idx + 1, static_cast<uint32_t>(kSQEBBAlignment));
        const auto num_final_wqebbs = new_head - final_sqebb_idx;
        for (int wqebb_idx = 0; wqebb_idx < num_final_wqebbs; ++ wqebb_idx) {
            const auto nop_sqebb_idx = final_sqebb_idx + wqebb_idx;
            const auto owner = static_cast<uint64_t>((nop_sqebb_idx & handle::kSQDepth) == 0) << 31;
            // 只有最后一条 NOP 请求完成与强序，其余 NOP 只是补齐 head 到 kSQEBBAlignment。
            const auto flags = wqebb_idx + 1 == num_final_wqebbs ? (2ULL << 16) | (1ULL << 21) : 0;
            const auto gm_nop_wqe = reinterpret_cast<__gm__ uint64_t*>(sq_base_addr +
                (nop_sqebb_idx & (handle::kSQDepth - 1)) * handle::kNumWQEBBBytes);
            AscendC::WriteGmByPassDCache(gm_nop_wqe,
                static_cast<uint64_t>(handle::kNopSQEHeaderTemplate | flags | owner |
                                      (nop_sqebb_idx & (handle::kSQDepth - 1))));
        }
        jetty.advance_sq(new_head - static_cast<uint32_t>(ub->jetty_info.packed_head));
        jetty.ring_doorbell();
    }

    if constexpr (kDoBarrier)
        comm::scalar::barrier<
            kNumRanks, kNumTimeoutCycles, true, false, true, kNumVecCores, kNumVecCores>(
                signals.common_signals, vec_core_idx, jetty_ptrs);
}

} // namespace deep_ep
