# AiDotNet vs PyTorch — Benchmark Comparison Report

**Source files:**
- `AiDotNet-test-result.json` (framework: AiDotNet 0.204.0.0, .NET runtime 10.0.9)
- `PyTorch-test-result` (framework: PyTorch / torch 2.12.0+cpu, Python 3.14.5, device: `cpu`, `torch_num_threads`: 8, `cuda_available`: false) — note: this file has no `.json` extension on disk, but its content is valid JSON.

Both files benchmark the same four model types — **MLP, CNN, LSTM, Transformer** — covering training timing, training memory, and inference latency/throughput/memory at batch sizes 1, 8, 32, and 128.

---

## Executive Summary

- **Inference (measured):** AiDotNet reports lower average latency and higher throughput than PyTorch in 15 of 16 model/batch-size combinations. The single exception is the Transformer at batch 128, where PyTorch is faster (11,216 vs 9,231 samples/s) — though the two Transformer models differ ~2× in parameter count.
- **Training (measured):** Mixed. AiDotNet completes total training faster for MLP and CNN; PyTorch is faster for LSTM (~2.8×) and Transformer. PyTorch reports lower average gradient-computation time for **all four** models.
- **Memory (measured, with caveats):** PyTorch's reported peak memory is far lower across the board (~327–365 MB vs ~870–2,274 MB for AiDotNet). However, the metric names differ (`rss_mb_peak` vs `managedRssMbPeak`/`memoryMbPeak`), so the measurements may not be directly comparable.
- **Accuracy/quality, stability/error rates:** **Not present in either file.** No comparison is possible.
- **Statistical significance:** **Cannot be assessed.** The files contain only point values (averages, p95s, and three per-epoch times); no standard deviations, confidence intervals, or run counts are recorded.

---

## 1. Benchmark Setup (as recorded in the files)

| Field | AiDotNet | PyTorch |
|---|---|---|
| Framework version | 0.204.0.0 | torch 2.12.0+cpu |
| Runtime | .NET 10.0.9 | Python 3.14.5 |
| Device | *not recorded* | `cpu` |
| Threads | *not recorded* | 8 (`torch_num_threads`) |
| CUDA / GPU | `nvidiaSmiSample: null` | `cuda_available: false`; GPU fields `null` |
| Backend label | `AiDotNetNeuralNetwork` | per-model `device: "cpu"` |

The AiDotNet file also contains a `neuralTypeProbe` array (25 AiDotNet type names confirming the library loaded). It contains no performance data.

### Model parameter counts — **not identical between frameworks**

| Model | AiDotNet params | PyTorch params | Ratio (ADN/PT) |
|---|---:|---:|---:|
| MLP | 468,874 | 468,874 | 1.00 |
| CNN | 20,490 | 9,930 | 2.06 |
| LSTM | 24,832 | 25,738 | 0.96 |
| Transformer | 35,520 | 69,706 | 0.51 |

Only the MLP is an exact like-for-like comparison by parameter count. CNN and Transformer results compare models of materially different sizes.

---

## 2. Training Performance (measured)

All values exactly as recorded. 3 epochs per model in both files.

| Model | Metric | AiDotNet | PyTorch | Faster |
|---|---|---:|---:|---|
| MLP | Total seconds | 0.705032 | 1.187479 | AiDotNet |
| MLP | Gradient sec (avg) | 0.011071 | 0.004132 | PyTorch |
| CNN | Total seconds | 2.040952 | 2.846833 | AiDotNet |
| CNN | Gradient sec (avg) | 0.033528 | 0.025640 | PyTorch |
| LSTM | Total seconds | 4.713862 | 1.692777 | PyTorch |
| LSTM | Gradient sec (avg) | 0.077923 | 0.014780 | PyTorch |
| Transformer | Total seconds | 6.047759 | 4.050657 | PyTorch |
| Transformer | Gradient sec (avg) | 0.099984 | 0.030188 | PyTorch |

Per-epoch times (seconds):

| Model | AiDotNet epochs | PyTorch epochs |
|---|---|---|
| MLP | 0.229, 0.224, 0.252 | 0.272, 0.318, 0.524 |
| CNN | 0.743, 0.686, 0.612 | 0.981, 0.979, 0.800 |
| LSTM | 1.604, 1.400, 1.710 | 0.805, 0.406, 0.418 |
| Transformer | 2.603, 1.799, 1.646 | 1.533, 1.220, 1.225 |

Data loading averages are small and similar in both files (0.000484–0.000807 s AiDotNet; 0.000515–0.000718 s PyTorch).

### Training memory (caveat: differing metric names)

| Model | AiDotNet `managedRssMbPeak` | PyTorch `rss_mb_peak` |
|---|---:|---:|
| MLP | 1,009.527 | 342.418 |
| CNN | 1,069.434 | 342.684 |
| LSTM | 1,721.246 | 341.742 |
| Transformer | 2,100.805 | 364.770 |

PyTorch additionally records `cpu_percent_avg` during training (MLP 59.4, CNN 156.6, LSTM 136.8, Transformer 283.7). AiDotNet has no CPU-utilization field, so this cannot be compared.

---

## 3. Inference Performance (measured)

### Throughput (samples/second) — higher is better

| Model | Batch | AiDotNet | PyTorch | Ratio (ADN/PT) |
|---|---:|---:|---:|---:|
| MLP | 1 | 5,610.256 | 3,506.410 | 1.60 |
| MLP | 8 | 31,083.654 | 13,457.026 | 2.31 |
| MLP | 32 | 41,455.019 | 22,262.078 | 1.86 |
| MLP | 128 | 133,943.474 | 17,075.546 | 7.84 |
| CNN | 1 | 1,974.766 | 622.288 | 3.17 |
| CNN | 8 | 5,577.004 | 3,294.116 | 1.69 |
| CNN | 32 | 6,191.374 | 3,765.228 | 1.64 |
| CNN | 128 | 8,539.313 | 7,408.787 | 1.15 |
| LSTM | 1 | 2,431.280 | 1,342.120 | 1.81 |
| LSTM | 8 | 7,629.962 | 2,530.068 | 3.02 |
| LSTM | 32 | 15,732.570 | 7,203.181 | 2.18 |
| LSTM | 128 | 27,145.830 | 19,137.425 | 1.42 |
| Transformer | 1 | 1,310.678 | 992.414 | 1.32 |
| Transformer | 8 | 4,929.712 | 3,931.384 | 1.25 |
| Transformer | 32 | 8,276.745 | 6,760.832 | 1.22 |
| Transformer | 128 | 9,230.659 | 11,215.973 | 0.82 |

### Average steady-state latency (ms) — lower is better

| Model | Batch | AiDotNet avg / p95 | PyTorch avg / p95 |
|---|---:|---|---|
| MLP | 1 | 0.178 / 0.225 | 0.285 / 0.634 |
| MLP | 8 | 0.257 / 0.359 | 0.594 / 1.819 |
| MLP | 32 | 0.772 / 1.706 | 1.437 / 4.399 |
| MLP | 128 | 0.956 / 1.524 | 7.496 / 20.385 |
| CNN | 1 | 0.506 / 0.791 | 1.607 / 2.717 |
| CNN | 8 | 1.434 / 2.944 | 2.429 / 4.896 |
| CNN | 32 | 5.168 / 6.400 | 8.499 / 19.342 |
| CNN | 128 | 14.989 / 18.090 | 17.277 / 24.157 |
| LSTM | 1 | 0.411 / 0.637 | 0.745 / 1.500 |
| LSTM | 8 | 1.048 / 1.729 | 3.162 / 6.451 |
| LSTM | 32 | 2.034 / 4.286 | 4.442 / 8.764 |
| LSTM | 128 | 4.715 / 6.069 | 6.688 / 12.494 |
| Transformer | 1 | 0.763 / 1.441 | 1.008 / 2.441 |
| Transformer | 8 | 1.623 / 1.949 | 2.035 / 3.261 |
| Transformer | 32 | 3.866 / 4.598 | 4.733 / 6.241 |
| Transformer | 128 | 13.867 / 23.383 | 11.412 / 14.771 |

AiDotNet's average latency is lower in 15 of 16 cells; the exception is Transformer at batch 128 (13.867 ms vs 11.412 ms).

### Inference peak memory (MB) — caveat: differing metric names

| Model | AiDotNet range (batch 1→128) | PyTorch range (batch 1→128) |
|---|---|---|
| MLP | 870.305 – 875.602 | 335.637 – 332.289 |
| CNN | 1,048.035 – 1,058.797 | 327.508 – 342.660 |
| LSTM | 1,642.020 – 1,705.191 | 327.637 – 331.207 |
| Transformer | 1,706.996 – 2,273.535 | 336.703 – 349.488 |

---

## 4. Accuracy / Quality Metrics

**No accuracy, loss, convergence, or any model-quality metric exists in either file.** No quality comparison can be made from this data.

## 5. Stability / Error Rates

**No error counts, failure records, retry data, or stability metrics exist in either file.** Both files contain complete result sets for all 4 models × 4 batch sizes, which indicates no run failed in a way that suppressed output, but nothing more can be concluded.

---

## 6. Strengths, Weaknesses, Trends, Anomalies

> The items below are **interpretations** of the measured values above, limited to patterns directly visible in the data.

### AiDotNet — observed strengths
- Lower inference latency and higher throughput in 15/16 configurations, including the parameter-matched MLP (up to 7.84× throughput at batch 128).
- Faster total training time for MLP and CNN — and the CNN result is achieved with ~2× the parameters of the PyTorch CNN.
- Lower p95 latency in 15/16 configurations, with notably tighter tails on MLP (e.g., 1.524 ms vs 20.385 ms at batch 128).

### AiDotNet — observed weaknesses
- Higher gradient-computation time in all 4 models (1.3×–5.3× PyTorch's).
- Slower total training for LSTM (4.714 s vs 1.693 s) and Transformer (6.048 s vs 4.051 s) — though its Transformer also has ~half the parameters of PyTorch's, which makes its slower training more notable.
- Reported peak memory 2.4×–6.5× higher than PyTorch's in every configuration (subject to the metric-comparability caveat).
- Memory grows with model/batch (e.g., Transformer training peak 2,100.8 MB; inference peak rising 1,707.0 → 2,273.5 MB across batch sizes), while PyTorch stays in a narrow ~327–365 MB band.

### PyTorch — observed strengths
- Consistently low and flat memory footprint (~327–365 MB across all training and inference runs, as reported).
- Fastest gradient computation in all 4 models.
- Faster LSTM and Transformer training; wins the only large-batch Transformer inference cell (11,216 vs 9,231 samples/s at batch 128) despite having ~2× the Transformer parameters.

### PyTorch — observed weaknesses
- Lower inference throughput and higher latency in 15/16 configurations.
- Larger latency tails: p95/avg ratios frequently exceed 2× (e.g., MLP batch 128: 20.385 ms p95 vs 7.496 ms avg).

### Anomalies and notable trends
1. **PyTorch MLP throughput is non-monotonic in batch size:** 22,262 samples/s at batch 32 drops to 17,076 at batch 128. All other model/framework series increase monotonically with batch size.
2. **PyTorch CNN warmup is non-monotonic:** batch 32 warmup (0.019884 s) exceeds batch 128 warmup (0.016238 s).
3. **AiDotNet wins total training time for MLP/CNN despite losing on gradient time** in those same models. The files do not break down where the remaining per-epoch time goes, so the cause cannot be determined from this data.
4. **AiDotNet LSTM inference memory dips then rises** (1,642.0 → 1,636.5 → 1,648.4 → 1,705.2 MB across batches 1→128); a peak metric decreasing between runs suggests measurement noise or per-run reset, but the files do not document the methodology.

### Statistical significance
**Not assessable.** The data provides single aggregate values per cell (averages and p95s) without sample counts, variances, or repeated independent runs. The three recorded epoch times per model show visible run-to-run variation (e.g., PyTorch LSTM epochs: 0.805, 0.406, 0.418 s — ~2× spread), indicating that small differences between frameworks should not be treated as significant.

---

## 7. Data Integrity Notes

### File naming
- The PyTorch file is named `PyTorch-test-result` **without a `.json` extension** (the request referenced `PyTorch-test-result.json`). Its content parses as valid JSON.

### Missing fields
- **Both files:** no accuracy/quality metrics, no error/stability metrics, no dataset description, no epoch count field (3 epochs inferred from array length), no number of inference iterations, no hardware description (CPU model, RAM).
- **AiDotNet file:** no device field, no thread count, no CPU-utilization metric (PyTorch records `cpu_percent_avg`), no Python/OS-equivalent environment details beyond `dotNetRuntime`.
- **PyTorch file:** no equivalent of AiDotNet's `neuralTypeProbe` (informational only).
- **GPU data:** absent in both (`nvidiaSmiSample: null` in AiDotNet; `gpu_util_percent_avg`/`gpu_mem_mb_peak` null and `cuda_available: false` in PyTorch). This is a CPU-only comparison on the PyTorch side; AiDotNet's device is unrecorded.

### Inconsistent values / unsupported comparisons
- **Parameter counts differ for 3 of 4 models** (CNN: 20,490 vs 9,930; LSTM: 24,832 vs 25,738; Transformer: 35,520 vs 69,706). CNN and Transformer comparisons are between materially different model sizes; only the MLP (468,874 = 468,874) is parameter-matched.
- **Memory metrics use different names** — `managedRssMbPeak`/`memoryMbPeak` (AiDotNet) vs `rss_mb_peak`/`memory_mb_peak` (PyTorch). Whether both measure the same quantity (process RSS vs managed heap) is not documented in the files; the memory comparison should be treated as indicative, not definitive.
- **AiDotNet's execution device is unrecorded**, so it cannot be confirmed that both frameworks ran on the same hardware/configuration.
- **Statistical comparison unsupported:** no variance, sample size, or repeated-run data in either file.

### Malformed records
- None found. Both files are syntactically valid JSON with internally consistent structure (4 models × training block × 4 inference batch sizes each).

---

## 8. Conclusion

Within the limits of the recorded data:

- **AiDotNet** is the faster framework for **inference** in this benchmark — lower average and p95 latency and higher throughput in 15 of 16 configurations, including the only parameter-matched model (MLP). It also posts faster total training times for MLP and CNN.
- **PyTorch** is consistently more **memory-efficient as reported** (roughly 2.4×–6.5× lower peak memory everywhere), has faster gradient computation in all four models, faster LSTM/Transformer training, and wins the batch-128 Transformer inference case.
- The comparison is weakened by three factors: unequal model sizes for CNN and Transformer, possibly non-equivalent memory metrics, and the absence of any variance/repetition data — so no difference here can be called statistically significant.
- **No conclusion about model accuracy, output quality, or stability is possible**, because neither file contains those metrics.
