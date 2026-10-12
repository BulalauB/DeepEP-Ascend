#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

namespace deep_ep::layout {

// Dispatch wire record（hidden 按调用方 dtype 原字节，kernel 不量化）：
//   [hidden][对齐 SF（若有）][对齐 int64 topk IDs（非 cached）][float topk weights][int source-global-ID（非 cached）]
// 整个 metadata 区再按 UB alignment 补齐；cached 省略 IDs/source-ID，但仍携带新 SF/weights。
// Combine 使用另一种 record：BF16 hidden + padded float weights，不发送 top-k IDs/source ID，目的地由 handle 恢复。
struct TokenLayout {
    // 字节数示例：FP8 H7168、SF=224B、topk=6、非cached：
    // ID区 align(6*8,32)=64B，metadata align(64+6*4+4,32)=96B，record=7168+224+96=7488B。
    // cached 去掉ID/source，metadata仅align(24,32)=32B，record=7424B；hidden/SF仍逐轮传输。
    // 相同H的combine则为BF16 hidden14336B + weight32B =14368B，不复用dispatch record stride。
    int num_hidden_bytes;
    int num_sf_bytes;
    bool with_metadata;
    int num_topk;
    int num_metadata_bytes;
    uint8_t* base;

    constexpr __aicore__ __forceinline__ TokenLayout(
        const int& num_hidden_bytes, const int& num_sf_bytes, const int& num_topk,
        const bool& with_metadata = true, uint8_t* base = nullptr):
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        with_metadata(with_metadata),
        num_topk(num_topk),
        // weight 长度固定为 topk*sizeof(float)；with_metadata 额外加 aligned ID 区与 source-global-ID。
        num_metadata_bytes(num_topk * sizeof(float) +
                           (with_metadata ? math::align<int>(num_topk * sizeof(int64_t), kNumUbAlignmentBytes) + sizeof(int) : 0)),
        base(base) {
        // TODO(HUAWEI): doing device assertions will cause hardware errors
        // EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        // EP_DEVICE_ASSERT(num_hidden_bytes % kNumUbAlignmentBytes == 0);
    }

    constexpr __aicore__ __forceinline__ TokenLayout(
        const int& num_hidden_bytes, const int& num_topk, uint8_t* base = nullptr):
        TokenLayout(num_hidden_bytes, 0, num_topk, true, base) {}

    // skip_hidden=true 返回网络第二个 SGE 的 metadata 总字节数（含 SF），不是跳过全部量化/路由信息。
    constexpr __aicore__ __forceinline__ int64_t get_num_bytes(const bool& skip_hidden = false) const {
        if (num_sf_bytes > 0) {
            return (skip_hidden ? 0 : num_hidden_bytes) +
                   math::align(num_sf_bytes, kNumUbAlignmentBytes) +
                   math::align(num_metadata_bytes, kNumUbAlignmentBytes);
        }
        return (skip_hidden ? 0 : num_hidden_bytes) + math::align(num_metadata_bytes, kNumUbAlignmentBytes);
    }

    template <typename dtype_t = uint8_t>
    constexpr __aicore__ __forceinline__ dtype_t* get_base_ptr(const bool& skip_hidden = false) const {
        return math::advance_ptr<dtype_t>(base, skip_hidden ? num_hidden_bytes : 0);
    }

    constexpr __aicore__ __forceinline__ void set_base_ptr(uint8_t* ptr) {
        base = ptr;
    }

    template <typename dtype_t = uint8_t>
    constexpr __aicore__ __forceinline__ dtype_t* get_hidden_ptr() const {
        return math::advance_ptr<dtype_t>(base, 0);
    }

    template <typename dtype_t = sf_pack_t>
    constexpr __aicore__ __forceinline__ dtype_t* get_sf_ptr() const {
        return math::advance_ptr<dtype_t>(base, math::align(num_hidden_bytes, kNumUbAlignmentBytes));
    }

    template <typename dtype_t = int64_t>
    constexpr __aicore__ __forceinline__ dtype_t* get_topk_idx_ptr() const {
        if (num_sf_bytes > 0)
            return math::advance_ptr<dtype_t>(get_sf_ptr(), math::align(num_sf_bytes, kNumUbAlignmentBytes));
        return math::advance_ptr<dtype_t>(base, num_hidden_bytes);
    }

    template <typename dtype_t = float>
    constexpr __aicore__ __forceinline__ dtype_t* get_topk_weights_ptr() const {
        return math::advance_ptr<dtype_t>(
            get_topk_idx_ptr(),
            with_metadata ? math::align<int>(sizeof(int64_t) * num_topk, kNumUbAlignmentBytes) : 0
        );
    }

    template <typename dtype_t = int>
    constexpr __aicore__ __forceinline__ dtype_t* get_src_token_global_idx_ptr() const {
        return math::advance_ptr<dtype_t>(get_topk_weights_ptr(), sizeof(float) * num_topk);
    }
};

struct BufferLayout {
    TokenLayout token_layout;
    int num_ranks;
    int num_max_tokens_per_rank;
    void* base;

    // TODO: the base's type may be `__ubuf__` or `__gm__`, try to add a template here
    constexpr __aicore__ __forceinline__ BufferLayout(
        const TokenLayout& token_layout,
        const int& num_ranks,
        const int& num_max_tokens_per_rank,
        void* base = nullptr):
        token_layout(token_layout),
        num_ranks(num_ranks),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    constexpr __aicore__ __forceinline__ int64_t get_num_bytes_per_token() const {
        return token_layout.get_num_bytes();
    }

    constexpr __aicore__ __forceinline__ int64_t get_num_bytes_per_rank() const {
        return static_cast<int64_t>(num_max_tokens_per_rank) * get_num_bytes_per_token();
    }

    constexpr __aicore__ __forceinline__ int64_t get_num_bytes() const {
        return static_cast<int64_t>(num_ranks) * get_num_bytes_per_rank();
    }

    constexpr __aicore__ __forceinline__ void* get_buffer_end_ptr() const {
        return math::advance_ptr<void>(base, get_num_bytes());
    }

    // source rank 分区 stride = Tmax*token_bytes；接收时该 source 的去重 token slots 在此固定区间内压紧。
    constexpr __aicore__ __forceinline__ BufferLayout get_rank_buffer(const int& rank_idx) const {
        return BufferLayout(
            token_layout, 1, num_max_tokens_per_rank,
            math::advance_ptr<uint8_t>(base, get_num_bytes_per_rank() * rank_idx));
    }

    constexpr __aicore__ __forceinline__ TokenLayout get_token_buffer(const int& token_idx = 0) const {
        auto token_layout_with_base = token_layout;
        token_layout_with_base.set_base_ptr(
            math::advance_ptr<uint8_t>(base, get_num_bytes_per_token() * token_idx));
        return token_layout_with_base;
    }
};

}  // namespace deep_ep::layout
