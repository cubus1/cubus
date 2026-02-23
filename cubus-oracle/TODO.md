# Prompt 11: Recognition as Projection — Test Checklist

## Projector64K

- [x] `project()` returns exactly 1024 u64s (65536 bits) — `test_projector_fingerprint_size`
- [x] Same embedding + same seed → same bits (deterministic) — `test_projector_deterministic`
- [x] Two identical embeddings: Hamming distance = 0 — `test_projector_self_similarity`
- [x] Two orthogonal embeddings: Hamming distance ≈ 32768 — `test_projector_orthogonal_vectors_half`
- [x] `project_signed()` direct i8 dot product (no f32 alloc) — refactored
- [x] `project_batch()` matches individual `project()` — `test_projection_batch`
- [x] Hyperplanes stored in contiguous flat buffer for cache locality — refactored

## Recognizer

- [x] `register_class` creates correct number of slots — `test_recognizer_register_classes`
- [x] `learn()` updates template toward sample — `test_recognizer_learn_improves`
- [x] `recognize_hamming()`: top-1 correct on training samples — `test_recognizer_self_recognition_hamming`
- [x] `recognize_orthogonal()`: top-1 correct on training samples — `test_recognizer_self_recognition_projection`
- [x] `recognize_two_stage()`: top-1 correct on training samples — `test_recognizer_self_recognition_two_stage`
- [x] 32 classes self-recognition > 80% — `test_recognizer_many_classes`

## Recognition Quality (Sweep: 72 configs)

- [x] All configs 100% top-1 (D={512,1024,2048}, K={4,8,16,32}, noise={0.3,0.5,1.0})
- [x] All three methods (Hamming, Projection, Two-Stage) tied at 100%
- [x] 100% novelty detection on random queries
- [ ] Harder configs needed: K=100+, noise=2.0+, to differentiate methods

## Novelty Detection

- [x] Known samples: low residual energy (0.025–0.43 depending on noise)
- [x] Random queries: residual > 0.5 — `test_novelty_detection_random_query`
- [x] 100% novelty detection rate across all sweep configs

## Timing (Sapphire Rapids, AVX-512, `-C target-cpu=native`)

- [ ] Single projection (64K bits): 56ms (target < 10ms — **needs SIMD**)
- [x] Single recognize (projection, 20 classes): **21us** (target < 1ms)
- [x] Full sweep (72 configs, 4K planes + blackboard): completes in ~30s
- [ ] Batch projection (100 × 64K): 5.6s (target < 500ms — **needs SIMD**)
- [x] Hamming distance (1024 u64s): **184ns** (fast)

## Performance Improvements Applied

- [x] Flat contiguous hyperplane buffer (was `Vec<Vec<f32>>`)
- [x] Direct i8 projection path (no Vec<f32> allocation per call)
- [x] Blackboard pattern: one `Projector64K` per dimensionality in sweep
- [x] `from_blackboard()` / `write_to_blackboard()` for arena-based sharing
- [x] `with_projector()` / `take_projector()` for ownership transfer
- [x] `run_recognition_sweep_fast()` — 4K planes for iteration speed

## Remaining Bottleneck

The projection hot loop (65536 dot products × D floats) is scalar. With AVX-512:
- 16 floats per vector register
- FMA: fused multiply-add
- Could process 16 hyperplane-dimension pairs per cycle
- Expected speedup: ~8-12× → projection from 56ms to ~5-7ms

## Architecture Notes

- `Projector64K` stores `hyperplanes_flat: Vec<f32>` — `[num_planes * d]` contiguous
- `Recognizer` owns projector, WAL, container, fingerprints, class averages
- `with_projector()` constructor for blackboard reuse
- `take_projector()` returns owned projector for transfer
- Sweep uses `Blackboard` from `rustynum-core` — 64-byte aligned arena
- `SplitMix64` from `rustynum-core` (consolidated PRNG)
- 26 passing tests in `recognize::tests`
- 164 total tests across all modules
