#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/common/exception.hpp>

#include <simt_api/device_warp_functions.h>

#ifdef __CCE__
namespace deep_ep::simt {

template <uint32_t kNumCycles = 300>
__simt_callee__ __forceinline__ void nop() {
    EP_STATIC_ASSERT(kNumCycles % 15 == 0);
    #pragma unroll 1
    for (int i = 0; i < kNumCycles / 15; ++ i)
        __nop();
}

__simt_callee__ __forceinline__ int exchange(const int& value, const int& lane_idx) {
    return asc_shfl(value, lane_idx);
}

template <int kMaxValue>
__simt_callee__ __forceinline__ bool deduplicate(const int& value, const int& lane_idx, const uint32_t lane_mask = 0xffffffffu) {
    // 以 lane_mask 限定同一个 token 的 top-k 子组，peers 初始为组内所有 lane。
    // 每次 ballot 区分 value 的一个 bit，保留与本 lane 全部 bit 一致的 lanes。
    // 例如 rank=[2,5,2,2]，三个 rank2 lane 最终 peers 相同；下面只保留最后一个 rank2 lane。
    // 此 helper 按 value 去重；dispatch 传入的是 rank，所以不合并不同 token，也不按专家 ID 去重。
    auto peers = lane_mask;
    #pragma unroll
    for (int i = 0; (1 << i) <= kMaxValue; ++ i) {
        const auto current_bit = (static_cast<unsigned>(value) >> i) & 1u;
        const auto ones = asc_ballot(current_bit);
        peers &= ones ^ (current_bit - 1u);
    }
    // 当前 lane 是相同 value 的最高 lane 时，右移后只剩自己这一位；保留最后一个匹配者而非第一个。
    // 无效 -1 也有一个代表，但调用方 dst_rank_idx>=0 的检查会过滤它。
    return (peers >> lane_idx) == 1u;
}

__simt_callee__ __forceinline__ int warp_inclusive_sum(int value, const int& lane_idx) {
    #pragma unroll
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        const auto synced = asc_shfl_up(value, offset);
        if (lane_idx >= offset)
            value += synced;
    }
    return value;
}

__simt_callee__ __forceinline__ int warp_exclusive_sum(const int& value, const int& lane_idx) {
    return warp_inclusive_sum(value, lane_idx) - value;
}

}  // namespace deep_ep::simt
#endif
