#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "llama-kv-tiered.h"

#include <memory>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // Sprint 4c step 3c-1: post-compute hook. Flushes TQ observation
    // queue; safe to call on non-TQ caches (no-op).
    void post_compute() override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // Sprint 4c step 2a: these return what the *user* requested (which may
    // be GGML_TYPE_TQ_KV*), separate from what the underlying ggml tensor
    // was allocated as (always a packed type). type_k()/type_v() above
    // return the latter.
    ggml_type requested_type_k() const { return requested_type_k_; }
    ggml_type requested_type_v() const { return requested_type_v_; }

    // True when the user asked for a TurboQuant KV type. The tiered_cache
    // objects in tq_k_caches/tq_v_caches are populated only in this case.
    // Writes and reads through them arrive in step 2b and 3 respectively.
    bool is_tq() const {
        return llama_kv_tq::is_turboquant_kv_type(requested_type_k_)
            || llama_kv_tq::is_turboquant_kv_type(requested_type_v_);
    }

    // Sprint 4c step 3b: read-side machinery.
    //
    // Materialize a contiguous fp16 view of the K (or V) cache for one
    // layer, covering token positions [start_pos, start_pos + n_positions).
    // Writes row-major [n_embd_gqa, n_positions] fp16 bytes into `out`.
    //
    // - Non-TQ cache or missing tiered_cache for this layer: clears
    //   `out` and returns (so callers can safely unconditionally invoke).
    // - Range validation and per-token hot/cold dispatch happens in
    //   tiered_cache::materialize_fp16_rows.
    //
    // NOTE: this function is not yet called by the inference path.
    // Step 3c wires it into build_attn. Here it is a tested utility.
    void tq_materialize_fp16_k(int32_t il,
                               int start_pos,
                               int n_positions,
                               std::vector<uint16_t> & out) const;
    void tq_materialize_fp16_v(int32_t il,
                               int start_pos,
                               int n_positions,
                               std::vector<uint16_t> & out) const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<llama_kv_cells> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // Sprint 4c step 2a: TurboQuant-side shadow storage. When the user
    // requests a TQ KV type, the ggml tensor in `layers[il].k` stays fp16
    // (so existing code paths keep working unchanged) and these parallel
    // tiered_cache objects hold the TurboQuant-packed state. Empty in the
    // non-TQ case.
    ggml_type requested_type_k_ = GGML_TYPE_COUNT;
    ggml_type requested_type_v_ = GGML_TYPE_COUNT;
    std::vector<std::unique_ptr<llama_kv_tq::tiered_cache>> tq_k_caches;
    std::vector<std::unique_ptr<llama_kv_tq::tiered_cache>> tq_v_caches;

    // Sprint 4c step 2b: deferred-observation queue. apply_ubatch runs
    // BEFORE the forward pass, so it can't read the freshly-written K/V
    // values yet. We queue the ubatch here, then at the *start* of the
    // next apply_ubatch (by which time the previous forward pass has
    // completed and the data has landed in layers[il].k/v) we flush:
    // read the fp16 values via ggml_backend_tensor_get and push them
    // into tq_k_caches/tq_v_caches via add_token.
    //
    // Known limitation: the very last ubatch of a decode() call isn't
    // flushed until the *next* decode() call. Step 2c will add explicit
    // flush points (end-of-decode, before state_write, etc.).
    struct pending_tq_obs {
        std::vector<uint32_t> idxs;   // slot indices that were written, flat
        std::vector<uint32_t> strm;   // parallel stream index for each idx
    };
    std::vector<pending_tq_obs> tq_pending_;

    void tq_flush_pending_();   // defined in llama-kv-cache.cpp

    // Sprint 4c step 3c-1: opt-in runtime self-consistency validation.
    // When LLAMA_TQ_VALIDATE=1 was set at construction, tq_flush_pending_
    // reads each just-added token back via tiered_cache::read_token_k/v
    // and compares to the fp32 buffer we pushed. For hot tokens this
    // must be exact (cos = 1.0 up to fp16 rounding); cold tokens reflect
    // per-bit compression loss. Accumulated stats are reported when
    // tq_validate_count_k/v_ cross tq_validate_report_every_ threshold.
    bool     tq_validate_ = false;
    int      tq_validate_report_every_ = 64;
    int      tq_validate_count_k_      = 0;
    int      tq_validate_count_v_      = 0;
    double   tq_validate_sum_cos_k_    = 0.0;
    double   tq_validate_sum_cos_v_    = 0.0;
    float    tq_validate_min_cos_k_    = 1.0f;
    float    tq_validate_min_cos_v_    = 1.0f;

    // Sprint 4c step 3c-2a: read-side view validation. Opt in via
    // LLAMA_TQ_VIEW_VALIDATE=1. After each post-compute flush, for
    // every just-flushed (slot, stream) we read the fp16 cache row
    // directly and compare it against tiered_cache::materialize_fp16_rows
    // output for the same logical position. This is the gating evidence
    // for Step 3c-2b (replace get_k/get_v reads with materialized views):
    // if the comparison consistently agrees within the per-bit cosine
    // threshold, the swap is safe.
    bool     tq_view_validate_ = false;
    int      tq_view_validate_count_k_   = 0;
    int      tq_view_validate_count_v_   = 0;
    double   tq_view_validate_sum_cos_k_ = 0.0;
    double   tq_view_validate_sum_cos_v_ = 0.0;
    float    tq_view_validate_min_cos_k_ = 1.0f;
    float    tq_view_validate_min_cos_v_ = 1.0f;

    // Sprint 4c step 3c-2b precursor: cold-path runtime validation. Opt
    // in via LLAMA_TQ_COLD_VALIDATE=1. After each flush, for any pending
    // observation whose tiered_cache position has already crossed into
    // the cold tier (because the flush itself overflowed the hot
    // window), we compare the cold-tier-decompressed bytes against the
    // fp16 cache row at the same (slot, stream). This is the missing
    // half of the read-side gate: 3c-2a covers hot, this covers cold.
    bool     tq_cold_validate_ = false;
    int      tq_cold_validate_count_k_   = 0;
    int      tq_cold_validate_count_v_   = 0;
    double   tq_cold_validate_sum_cos_k_ = 0.0;
    double   tq_cold_validate_sum_cos_v_ = 0.0;
    float    tq_cold_validate_min_cos_k_ = 1.0f;
    float    tq_cold_validate_min_cos_v_ = 1.0f;

    // Sprint 4c step 3c-2b: opt-in write-back readthrough. Enabled by
    // LLAMA_TQ_READTHROUGH=1. After each post-compute flush, we
    // materialize all observed positions from tiered_cache and
    // ggml_backend_tensor_set them back into layers[il].k/v. Attention
    // on the *next* forward pass therefore reads decompressed-fp16
    // data (within the per-bit cosine bound) — exactly what step 3c-3
    // will see once the backbone shrinks. Lets us measure end-to-end
    // logit divergence before committing to memory layout changes.
    //
    // Limitations: this assumes n_stream == 1 and that tiered_cache
    // positions correspond 1:1 to slot indices (i.e. no seq_rm has
    // shuffled them). Both are true for plain decode loops; complex
    // sampling like beam search or seq_rm-driven workflows would need
    // an explicit slot->pos mapping (deferred to 3c-3).
    bool     tq_readthrough_ = false;
    void     tq_apply_readthrough_();

    // Per-stream slot → most-recent tiered_cache position. -1 means never
    // observed (or invalidated by clear/seq_rm). Updated on each observe;
    // overwriting is intentional (slot reuse → latest mapping wins).
    // Indexing: tq_slot_to_pos_k_[strm][slot] -> tc_pos.
    std::vector<std::vector<int32_t>> tq_slot_to_pos_k_;
    std::vector<std::vector<int32_t>> tq_slot_to_pos_v_;

    // Invalidate the slot map for a single stream. Called from
    // seq_rm/seq_cp/seq_keep so the writeback doesn't restore data
    // for slots that have been logically removed.
    void tq_invalidate_stream_(uint32_t strm);

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
