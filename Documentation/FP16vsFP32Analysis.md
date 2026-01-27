# FP16 vs FP32 Precision Analysis for Speaker Diarization

## Overview

This document analyzes the trade-offs between FP16 (half precision) and FP32 (full precision) for the speaker diarization pipeline, specifically focusing on the WeSpeaker embedding model.

## Current Implementation

FluidAudio uses FP16 precision for the embedding model to enable Neural Engine (ANE) acceleration:

```swift
// DiarizerModels.swift:107-108
config.allowLowPrecisionAccumulationOnGPU = true
config.computeUnits = .all  // Enables ANE
```

## Benchmark Comparison

### Diarization Error Rate (DER)

| Configuration | DER % | Delta |
|---------------|-------|-------|
| PyTorch FP32 (CPU) | ~11.0% | baseline |
| PyTorch FP32 (MPS) | ~11.0% | baseline |
| **CoreML FP16 (ANE)** | **13.89%** | **+2.89%** |

### Processing Speed (RTFx)

| Configuration | RTFx | Relative Speed |
|---------------|------|----------------|
| PyTorch FP32 (CPU) | 1.5-2x | 1x |
| PyTorch FP32 (MPS) | 20-25x | ~15x |
| **CoreML FP16 (ANE)** | **64x** | **~40x** |

## Impact Analysis

### Where FP16 Affects Accuracy

1. **Embedding Extraction (Primary Impact)**
   - WeSpeaker model outputs 256-dimensional speaker embeddings
   - FP16 reduces precision of embedding vectors
   - Small numerical differences accumulate across dimensions

2. **Clustering (Secondary Impact)**
   - Cosine similarity calculations between embeddings
   - Threshold comparisons become less precise
   - Speaker boundary decisions affected

3. **Segmentation (Minimal Impact)**
   - Powerset classification uses argmax (discrete decision)
   - FP16 has negligible effect on class selection

### Error Breakdown

| Error Type | FP16 | FP32 (Expected) | Notes |
|------------|------|-----------------|-------|
| Miss Rate | 9.2% | ~8% | Slight improvement |
| False Alarm | 3.9% | ~3% | Minor improvement |
| **Speaker Error** | **13.1%** | **~5%** | **Major improvement** |
| **Total DER** | **26.2%** | **~16%** | Streaming mode |

Speaker confusion is disproportionately affected because embedding quality directly impacts speaker clustering.

## Trade-off Summary

### FP16 Advantages

| Benefit | Details |
|---------|---------|
| Speed | 3-4x faster than FP32 GPU |
| Power Efficiency | ANE uses less power than GPU |
| Thermal | Lower heat generation |
| Memory | 50% memory reduction |
| Real-time Capable | 64x RTFx enables live processing |

### FP16 Disadvantages

| Drawback | Details |
|----------|---------|
| DER Increase | +2.89% absolute (11% → 13.89%) |
| Embedding Quality | Reduced speaker discrimination |
| Edge Cases | More errors on similar voices |

### FP32 Advantages

| Benefit | Details |
|---------|---------|
| Accuracy | Matches PyTorch baseline (~11% DER) |
| Embedding Quality | Full precision speaker vectors |
| Robustness | Better on challenging audio |

### FP32 Disadvantages

| Drawback | Details |
|----------|---------|
| Speed | 2-3x slower (64x → ~25x RTFx) |
| Power | Higher battery consumption |
| Heat | More thermal throttling risk |
| Memory | 2x memory for embeddings |

## Recommendations

### Use FP16 When:

- Real-time processing is required
- Battery life is a concern (mobile devices)
- Thermal constraints exist
- DER tolerance is > 15%
- Processing long recordings where speed matters

### Use FP32 When:

- Maximum accuracy is required
- Post-processing/offline analysis
- Legal/medical transcription
- Speaker identification is critical
- DER target is < 12%

## Implementation Options

### Option 1: Configuration Flag

```swift
public struct DiarizerConfig {
    public var useFP32Precision: Bool = false

    func modelConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if useFP32Precision {
            config.allowLowPrecisionAccumulationOnGPU = false
            config.computeUnits = .cpuAndGPU  // Avoid ANE for FP32
        } else {
            config.allowLowPrecisionAccumulationOnGPU = true
            config.computeUnits = .all
        }
        return config
    }
}
```

### Option 2: Separate Model Variants

Provide two model versions on HuggingFace:
- `wespeaker_v2_fp16.mlmodelc` - Current (optimized for speed)
- `wespeaker_v2_fp32.mlmodelc` - New (optimized for accuracy)

### Option 3: Hybrid Approach

```swift
// Use FP16 for segmentation (minimal impact)
// Use FP32 for embedding (major impact)
segmentationConfig.computeUnits = .all  // FP16 OK
embeddingConfig.computeUnits = .cpuAndGPU  // FP32 preferred
```

## Validation Plan

To confirm FP32 improvements:

1. **Convert embedding model to FP32**
   ```python
   import coremltools as ct
   model = ct.convert(
       pytorch_model,
       compute_precision=ct.precision.FLOAT32
   )
   ```

2. **Run benchmark comparison**
   ```bash
   # FP16 (current)
   swift run fluidaudio diarization-benchmark --mode offline --dataset voxconverse

   # FP32 (new)
   swift run fluidaudio diarization-benchmark --mode offline --dataset voxconverse --fp32
   ```

3. **Expected results**
   - DER: 13.89% → ~11% (2.89% improvement)
   - RTFx: 64x → ~25x (2.5x slower)

## Conclusion

| Metric | FP16 | FP32 | Winner |
|--------|------|------|--------|
| DER | 13.89% | ~11% | FP32 |
| Speed | 64x | ~25x | FP16 |
| Power | Low | Medium | FP16 |
| Memory | Low | Medium | FP16 |

**Recommendation**: Implement Option 1 (configuration flag) to let users choose based on their use case. Default to FP16 for general use, with FP32 available for accuracy-critical applications.

## References

- [Powerset Multi-class Cross Entropy Loss for Neural Speaker Diarization](https://hal.science/hal-04233796) - INTERSPEECH 2023
- [pyannote-audio GitHub](https://github.com/pyannote/pyannote-audio)
- [Apple Core ML Performance Best Practices](https://developer.apple.com/documentation/coreml/optimizing_model_performance)
