#pragma once

#include <c_api/asc_simd.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>

#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include <kernel_operator.h>

namespace deep_ep {

// 共享归约器处理 UB 内的 BF16 hidden 行，既用于专家卡本地 partial，也用于源卡最终 rank partial 求和。
// 数值路径有特化：0 行写零；1 行复制；2 行且无 extra metadata 时直接 BF16 add；其余路径 FP32 累加再舍入。
// 因此不能笼统声明 combine 全程只舍入一次：本地 rank partial 与最终 source sum 是两个归约层次。
template <int kHidden, int kNumTopk, int kNumStrideElems, int kNumSlots = 0>
__simd_vf__ __forceinline__ void combine_reduce_vf(__ubuf__ bfloat16_t* base_ptr, __ubuf__ bfloat16_t* output_ptr,
                                                    const int count, const int slot_start_idx = 0) {
    static constexpr int kNumElemsPerFP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    EP_STATIC_ASSERT(kHidden % (kNumElemsPerFP32 * 2) == 0);
    EP_STATIC_ASSERT(kNumTopk >= 1);

    vector_bfloat16_t bf16_zero;
    asc_duplicate_scalar(bf16_zero, 0);

    // 没有任何有效 route/bias 时产生确定零输出，不能留下旧 UB 内容。
    if (count == 0) {
        #pragma unroll 1
        for (int offset = 0; offset < kHidden; offset += kNumElemsPerFP32 * 2)
            asc_storealign(output_ptr + offset, bf16_zero, asc_create_mask_b16(PAT_ALL));
        return;
    } else if (count == 1) {
        __ubuf__ bfloat16_t* ub_input_ptr = base_ptr + slot_start_idx * kNumStrideElems;
        #pragma unroll 1
        for (int offset = 0; offset < kHidden; offset += kNumElemsPerFP32 * 2) {
            vector_bfloat16_t bf16_reg;
            asc_loadalign_postupdate(bf16_reg, ub_input_ptr, kNumElemsPerFP32 * 2);
            asc_storealign(output_ptr + offset, bf16_reg, asc_create_mask_b16(PAT_ALL));
        }
        return;
    }

    // 两行 hidden-only 快路径直接 BF16 向量加法，不执行下面的 FP32 widening 分支。
    if (count == 2 and kNumStrideElems == kHidden) {
        const auto second_slot_idx = kNumSlots != 0 and slot_start_idx + 1 == kNumSlots ? 0 : slot_start_idx + 1;
        auto first_ptr = base_ptr + slot_start_idx * kNumStrideElems;
        auto second_ptr = base_ptr + second_slot_idx * kNumStrideElems;
        const auto mask = asc_create_mask_b16(PAT_ALL);
        #pragma unroll 2
        for (int offset = 0; offset < kHidden; offset += kNumElemsPerFP32 * 2) {
            vector_bfloat16_t first, second;
            asc_loadalign_postupdate(first, first_ptr, kNumElemsPerFP32 * 2);
            asc_loadalign_postupdate(second, second_ptr, kNumElemsPerFP32 * 2);
            asc_add(first, first, second, mask);
            asc_storealign(output_ptr + offset, first, mask);
        }
        return;
    }

    vector_bool u32_mask = asc_create_mask_b32(PAT_ALL);
    // 环中从 slot_start 到末尾为第一段，剩余输入从 slot 0 继续；GM 行次序与 UB 环物理连续性不同。
    const auto num_wrap_count = kNumSlots == 0 ? count : math::min(kNumSlots - slot_start_idx, count);
    #pragma unroll 1
    for (int offset = 0; offset < kHidden; offset += kNumElemsPerFP32 * 2) {
        vector_bfloat16_t bf16_regs[2];
        vector_float acc_regs[2];
        __ubuf__ bfloat16_t* ptr = base_ptr + slot_start_idx * kNumStrideElems + offset;
        asc_loadalign_postupdate(bf16_regs[0], ptr, kNumStrideElems);
        // BF16 作为 FP32 高半字、低半字补零，实现无误差扩宽；随后逐行累加到两份 FP32 vector。
        asc_intlv(reinterpret_cast<vector_bfloat16_t&>(acc_regs[0]), reinterpret_cast<vector_bfloat16_t&>(acc_regs[1]), bf16_zero,
                  bf16_regs[0]);
        #pragma unroll 1
        for (int i = 1; i < num_wrap_count; ++ i) {
            vector_float fp32_regs[2];
            asc_loadalign_postupdate(bf16_regs[0], ptr, kNumStrideElems);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(fp32_regs[0]), reinterpret_cast<vector_bfloat16_t&>(fp32_regs[1]), bf16_zero,
                      bf16_regs[0]);
            asc_add(acc_regs[0], acc_regs[0], fp32_regs[0], u32_mask);
            asc_add(acc_regs[1], acc_regs[1], fp32_regs[1], u32_mask);
        }
        if constexpr (kNumSlots != 0) {
            ptr = base_ptr + offset;
            #pragma unroll 1
            for (int i = num_wrap_count; i < count; ++ i) {
                vector_float fp32_regs[2];
                asc_loadalign_postupdate(bf16_regs[0], ptr, kNumStrideElems);
                asc_intlv(reinterpret_cast<vector_bfloat16_t&>(fp32_regs[0]), reinterpret_cast<vector_bfloat16_t&>(fp32_regs[1]),
                          bf16_zero, bf16_regs[0]);
                asc_add(acc_regs[0], acc_regs[0], fp32_regs[0], u32_mask);
                asc_add(acc_regs[1], acc_regs[1], fp32_regs[1], u32_mask);
            }
        }

        asc_float2bfloat16_rn(bf16_regs[0], acc_regs[0], u32_mask);
        asc_float2bfloat16_rn(bf16_regs[1], acc_regs[1], u32_mask);
        asc_deintlv(bf16_regs[0], bf16_regs[1], bf16_regs[0], bf16_regs[1]);
        asc_storealign(output_ptr + offset, bf16_regs[0], asc_create_mask_b16(PAT_ALL));
    }
}

template <uint32_t kNumHiddenElems, uint32_t kNumTopk, uint32_t kNumMutex = 32, uint32_t kMutexOffset = 0,
          uint32_t kNumExtraBytes = 0>
// slot_idx 是下一次写入 UB 的槽号；每槽 mutex 仲裁 MTE2 写、Vector 读/改、MTE3 读。
// 这是片上 staging ring，不是 GM 接收区 ring；GM 返回 partial 仍有独占 token/subslot 空间。
struct RingReducer {
    static constexpr uint32_t kNumHiddenBytes = kNumHiddenElems * sizeof(bfloat16_t);
    static constexpr uint32_t kNumTokenElems =
        kNumHiddenElems + math::ceil_div<uint32_t>(kNumExtraBytes, sizeof(bfloat16_t));

    struct Buffer {
        bfloat16_t buf[kNumMutex][kNumTokenElems];
    };

    __ubuf__ Buffer* ub;
    uint32_t slot_idx;

    __aicore__ __forceinline__ RingReducer(__ubuf__ Buffer* ub): ub(ub), slot_idx(0) {}

    __aicore__ __forceinline__ void load_next_token(__gm__ bfloat16_t* ptr,
                                                    asc_load_l2_cache_mode mode = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                                                    const uint32_t num_extra_bytes = 0) {
        // MTE2 取得目标槽所有权后装载，unlock 发布该管线的槽操作；Vector/MTE3 后续 lock 建立消费顺序。
        // 不能以 Scalar 更新 slot_idx 当作 DMA 已经完成的证明。
        asc_lock(PIPE_MTE2, kMutexOffset + slot_idx);
        asc_copy_gm2ub_align(ub->buf[slot_idx], ptr, 1, kNumHiddenBytes + num_extra_bytes, 0, 0, true, mode, 0, 0);
        asc_unlock(PIPE_MTE2, kMutexOffset + slot_idx);
        slot_idx = slot_idx + 1 == kNumMutex ? 0 : slot_idx + 1;
    }

    __aicore__ __forceinline__ void load_first_2_token(__gm__ bfloat16_t* base, int idx_0, int idx_1,
                                                       asc_load_l2_cache_mode mode = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM) {
        // 两行 burst 必须占两个相邻 UB 槽；若下一槽是最后一槽，先绕到 0，避免第二行写出 ring 边界。
        slot_idx = slot_idx == kNumMutex - 1 ? 0 : slot_idx;
        const auto min_idx = idx_0 < idx_1 ? idx_0 : idx_1;
        // 将两行按较小 GM index 起点、绝对 stride 合成一次两 burst 读取；求和不要求这两行保持原 order。
        const auto stride = idx_0 < idx_1 ? idx_1 - idx_0 : idx_0 - idx_1;
        asc_lock(PIPE_MTE2, kMutexOffset + slot_idx);
        asc_lock(PIPE_MTE2, kMutexOffset + slot_idx + 1);
        asc_copy_gm2ub_align(ub->buf[slot_idx], base + static_cast<int64_t>(min_idx) * kNumTokenElems,
                            2, kNumHiddenBytes + kNumExtraBytes, 0, 0, true, mode,
                            static_cast<uint64_t>(stride) * (kNumHiddenBytes + kNumExtraBytes),
                            kNumHiddenBytes + kNumExtraBytes);
        asc_unlock(PIPE_MTE2, kMutexOffset + slot_idx);
        asc_unlock(PIPE_MTE2, kMutexOffset + slot_idx + 1);
        slot_idx = slot_idx + 2 == kNumMutex ? 0 : slot_idx + 2;
    }

    __aicore__ __forceinline__ void store_last_token(__gm__ bfloat16_t* output_addr,
                                                     asc_store_l2_cache_mode mode = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                                                     __gm__ uint8_t* extra_output_addr = nullptr,
                                                     const uint32_t num_extra_output_bytes = kNumExtraBytes) {
        // 结果存在最近使用的槽（slot_idx-1，含回绕）；MTE3 lock 保证该槽的读取晚于归约或单行 load。
        const auto output_slot_idx = slot_idx == 0 ? kNumMutex - 1 : slot_idx - 1;
        asc_lock(PIPE_MTE3, kMutexOffset + output_slot_idx);
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(output_addr),
                            reinterpret_cast<__ubuf__ uint8_t*>(ub->buf[output_slot_idx]),
                            1, kNumHiddenBytes, mode, 0, 0);
        if constexpr (kNumExtraBytes > 0) {
            if (extra_output_addr != nullptr) {
                asc_copy_ub2gm_align(extra_output_addr,
                    reinterpret_cast<__ubuf__ uint8_t*>(ub->buf[output_slot_idx]) + kNumHiddenBytes,
                    num_extra_output_bytes);
            }
        }
        asc_unlock(PIPE_MTE3, kMutexOffset + output_slot_idx);
    }

    __aicore__ __forceinline__ void reduce(const uint32_t num_slots) {
        // 下一空槽减 num_slots 得本次输入环区间；num_slots=0 仍锁一槽用于生成零输出。
        const auto start_slot_idx = slot_idx < num_slots ? kNumMutex - (num_slots - slot_idx) : slot_idx - num_slots;
        const auto output_slot_idx = num_slots > 0 ? (slot_idx == 0 ? kNumMutex - 1 : slot_idx - 1) : slot_idx;
        const auto num_locked_slots = math::max(1U, num_slots);
        for (uint32_t i = 0, input_slot_idx = start_slot_idx; i < num_locked_slots; ++ i) {
            asc_lock(PIPE_V, kMutexOffset + input_slot_idx);
            input_slot_idx = input_slot_idx + 1 == kNumMutex ? 0 : input_slot_idx + 1;
        }
        // 先锁全部本次输入，Vector 归约原地写最后一槽；unlock 后才能允许 MTE3 读取输出、MTE2 复用旧槽。
        combine_reduce_vf<kNumHiddenElems, kNumTopk, kNumTokenElems, kNumMutex>(
            ub->buf[0], ub->buf[output_slot_idx], num_slots, start_slot_idx);
        for (uint32_t i = 0, input_slot_idx = start_slot_idx; i < num_locked_slots; ++ i) {
            asc_unlock(PIPE_V, kMutexOffset + input_slot_idx);
            input_slot_idx = input_slot_idx + 1 == kNumMutex ? 0 : input_slot_idx + 1;
        }
        if (num_slots == 0)
            slot_idx = slot_idx + 1 == kNumMutex ? 0 : slot_idx + 1;
    }
};

}  // namespace deep_ep
