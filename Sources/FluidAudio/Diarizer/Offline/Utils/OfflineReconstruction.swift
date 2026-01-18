import Accelerate
import Foundation

struct OfflineReconstruction {
    private let config: OfflineDiarizerConfig
    private let logger = AppLogger(category: "OfflineReconstruction")

    private struct Accumulator {
        var start: Double
        var end: Double
        var scoreSum: Double
        var frameCount: Int
    }

    init(config: OfflineDiarizerConfig) {
        self.config = config
    }

    func buildSegments(
        segmentation: SegmentationOutput,
        hardClusters: [[Int]],
        centroids: [[Double]],
        timedEmbeddings: [TimedEmbedding] = []
    ) -> [TimedSpeakerSegment] {
        guard segmentation.numChunks > 0, segmentation.numFrames > 0 else { return [] }

        let frameDuration = segmentation.frameDuration
        guard frameDuration > 0 else { return [] }

        let clusterCount = max(centroids.count, 1)
        let gapThreshold = max(config.minGapDuration, config.segmentationMinDurationOff)

        var maxTime = 0.0
        for chunkIndex in 0..<segmentation.numChunks {
            let offset = chunkStartTime(for: chunkIndex, segmentation: segmentation)
            let end = offset + Double(segmentation.numFrames) * frameDuration
            if end > maxTime {
                maxTime = end
            }
        }

        let totalFrames = max(1, Int(ceil(maxTime / frameDuration)))
        var activationSums = Array(
            repeating: Array(repeating: 0.0, count: clusterCount),
            count: totalFrames
        )
        var activationCounts = Array(
            repeating: Array(repeating: 0.0, count: clusterCount),
            count: totalFrames
        )
        var expectedCountSums = [Double](repeating: 0, count: totalFrames)
        var expectedCountWeights = [Double](repeating: 0, count: totalFrames)

        for chunkIndex in 0..<segmentation.numChunks {
            guard chunkIndex < segmentation.speakerWeights.count else { continue }
            let chunkWeights = segmentation.speakerWeights[chunkIndex]
            guard !chunkWeights.isEmpty else { continue }

            let chunkOffset = chunkStartTime(for: chunkIndex, segmentation: segmentation)
            let chunkAssignments =
                chunkIndex < hardClusters.count
                ? hardClusters[chunkIndex] : Array(repeating: -2, count: segmentation.numSpeakers)

            for frameIndex in 0..<chunkWeights.count {
                let frameStart = chunkOffset + Double(frameIndex) * frameDuration
                var globalFrame = Int((frameStart / frameDuration).rounded())
                if globalFrame < 0 {
                    globalFrame = 0
                } else if globalFrame >= totalFrames {
                    globalFrame = totalFrames - 1
                }

                let weights = chunkWeights[frameIndex]
                var frameActivations = [Double](repeating: 0, count: clusterCount)

                for speakerIndex in 0..<min(weights.count, chunkAssignments.count) {
                    let cluster = chunkAssignments[speakerIndex]
                    guard cluster >= 0, cluster < clusterCount else { continue }
                    let value = Double(weights[speakerIndex])
                    if value > frameActivations[cluster] {
                        frameActivations[cluster] = value
                    }
                }

                let expectedCount = weights.reduce(0.0) { partialSum, value in
                    partialSum + Double(value)
                }
                expectedCountSums[globalFrame] += expectedCount
                expectedCountWeights[globalFrame] += 1

                for cluster in 0..<clusterCount {
                    let value = frameActivations[cluster]
                    if value > 0 {
                        activationSums[globalFrame][cluster] += value
                        activationCounts[globalFrame][cluster] += 1
                    }
                }
            }
        }

        var activationAverages = Array(
            repeating: Array(repeating: 0.0, count: clusterCount),
            count: totalFrames
        )
        for frame in 0..<totalFrames {
            let sums = activationSums[frame]
            let counts = activationCounts[frame]
            var averages = [Double](repeating: 0, count: clusterCount)

            // Vectorized division: averages = sums / counts (where counts > 0)
            sums.withUnsafeBufferPointer { sumsPtr in
                counts.withUnsafeBufferPointer { countsPtr in
                    averages.withUnsafeMutableBufferPointer { averagesPtr in
                        guard let sumsBase = sumsPtr.baseAddress,
                            let countsBase = countsPtr.baseAddress,
                            let averagesBase = averagesPtr.baseAddress
                        else { return }

                        vDSP_vdivD(
                            countsBase,
                            1,
                            sumsBase,
                            1,
                            averagesBase,
                            1,
                            vDSP_Length(clusterCount)
                        )
                    }
                }
            }

            // Zero out results where count was 0 (to avoid division by zero artifacts)
            for cluster in 0..<clusterCount where counts[cluster] == 0 {
                averages[cluster] = 0
            }

            activationAverages[frame] = averages
        }

        var speakerCountPerFrame = [Int](repeating: 0, count: totalFrames)
        var speakerCountHistogram: [Int: Int] = [:]
        let maxAllowedSpeakers = min(clusterCount, segmentation.numSpeakers)
        for frame in 0..<totalFrames {
            let weight = expectedCountWeights[frame]
            guard weight > 0 else { continue }
            var rounded = Int((expectedCountSums[frame] / weight).rounded(.toNearestOrEven))
            if rounded < 0 { rounded = 0 }
            if rounded > maxAllowedSpeakers { rounded = maxAllowedSpeakers }
            speakerCountPerFrame[frame] = rounded
            speakerCountHistogram[rounded, default: 0] += 1
        }

        if !speakerCountHistogram.isEmpty {
            let histogramString =
                speakerCountHistogram
                .sorted { $0.key < $1.key }
                .map { "\($0.key):\($0.value)" }
                .joined(separator: ", ")
            logger.debug("Speaker-count histogram \(histogramString)")
        }

        var perFrameClusters = [[Int]](repeating: [], count: totalFrames)
        for frame in 0..<totalFrames {
            let required = speakerCountPerFrame[frame]
            guard required > 0 else { continue }

            // Filter to clusters with activation > 0, then rank by activation strength
            let ranked = activationSums[frame].enumerated()
                .filter { $0.element > 0 }
                .sorted { $0.element > $1.element }

            // Select all active clusters (up to clusterCount) instead of just top-K
            // This respects VBx clustering results even when speakers don't overlap
            let selected = ranked.prefix(clusterCount).map { $0.offset }
            perFrameClusters[frame] = selected
        }

        // Debug: Log cluster frame counts for verification
        var clusterFrameCounts = [Int: Int]()
        for frameClusters in perFrameClusters {
            for cluster in frameClusters {
                clusterFrameCounts[cluster, default: 0] += 1
            }
        }
        if !clusterFrameCounts.isEmpty {
            let countsString = clusterFrameCounts
                .sorted { $0.key < $1.key }
                .map { "S\($0.key + 1):\($0.value)" }
                .joined(separator: ", ")
            logger.debug("Cluster frame counts: \(countsString)")
        }

        var activeSegments: [Int: Accumulator] = [:]
        var rawSegments: [TimedSpeakerSegment] = []

        for frameIndex in 0..<totalFrames {
            let frameStart = Double(frameIndex) * frameDuration
            let frameEnd = frameStart + frameDuration
            let activeClusters = Set(perFrameClusters[frameIndex])
            let averageScores = activationAverages[frameIndex]

            for (cluster, accumulator) in activeSegments where !activeClusters.contains(cluster) {
                appendSegment(
                    cluster: cluster,
                    accumulator: accumulator,
                    endTime: frameStart,
                    centroids: centroids,
                    timedEmbeddings: timedEmbeddings,
                    output: &rawSegments
                )
            }
            activeSegments = activeSegments.filter { activeClusters.contains($0.key) }

            for cluster in activeClusters {
                let score = averageScores.indices.contains(cluster) ? averageScores[cluster] : 0
                if var existing = activeSegments[cluster] {
                    existing.end = frameEnd
                    existing.scoreSum += score
                    existing.frameCount += 1
                    activeSegments[cluster] = existing
                } else {
                    activeSegments[cluster] = Accumulator(
                        start: frameStart,
                        end: frameEnd,
                        scoreSum: score,
                        frameCount: 1
                    )
                }
            }
        }

        for (cluster, accumulator) in activeSegments {
            appendSegment(
                cluster: cluster,
                accumulator: accumulator,
                endTime: accumulator.end,
                centroids: centroids,
                timedEmbeddings: timedEmbeddings,
                output: &rawSegments
            )
        }

        // Debug: Log raw segments per speaker before merge
        let rawSpeakerCounts = Dictionary(grouping: rawSegments, by: \.speakerId)
            .mapValues { $0.count }
        if !rawSpeakerCounts.isEmpty {
            let countsString = rawSpeakerCounts.sorted { $0.key < $1.key }
                .map { "\($0.key):\($0.value)" }
                .joined(separator: ", ")
            logger.debug("Raw segments per speaker: \(countsString)")
        }

        let merged = mergeSegments(rawSegments, gapThreshold: gapThreshold)

        // Debug: Log merged segments per speaker
        let mergedSpeakerCounts = Dictionary(grouping: merged, by: \.speakerId)
            .mapValues { $0.count }
        if !mergedSpeakerCounts.isEmpty {
            let countsString = mergedSpeakerCounts.sorted { $0.key < $1.key }
                .map { "\($0.key):\($0.value)" }
                .joined(separator: ", ")
            logger.debug("Merged segments per speaker: \(countsString)")
        }

        return sanitize(segments: merged)
    }

    func buildSpeakerDatabase(
        segments: [TimedSpeakerSegment]
    ) -> [String: [Float]] {
        var sums: [String: [Float]] = [:]
        var counts: [String: Int] = [:]

        for segment in segments {
            if var current = sums[segment.speakerId] {
                let embedding = segment.embedding
                precondition(
                    embedding.count == current.count,
                    "Embedding dimensionality mismatch while accumulating speaker database"
                )
                embedding.withUnsafeBufferPointer { sourcePointer in
                    current.withUnsafeMutableBufferPointer { destinationPointer in
                        guard
                            let sourceBase = sourcePointer.baseAddress,
                            let destinationBase = destinationPointer.baseAddress
                        else { return }
                        cblas_saxpy(
                            Int32(embedding.count),
                            1.0,
                            sourceBase,
                            1,
                            destinationBase,
                            1
                        )
                    }
                }
                sums[segment.speakerId] = current
            } else {
                sums[segment.speakerId] = segment.embedding
            }
            counts[segment.speakerId, default: 0] += 1
        }

        var database: [String: [Float]] = [:]
        for (speaker, sum) in sums {
            guard let count = counts[speaker], count > 0 else { continue }
            var averaged = sum
            var scale = 1 / Float(count)
            let length = averaged.count
            averaged.withUnsafeMutableBufferPointer { pointer in
                guard let baseAddress = pointer.baseAddress else { return }
                vDSP_vsmul(
                    baseAddress,
                    1,
                    &scale,
                    baseAddress,
                    1,
                    vDSP_Length(length)
                )
            }
            database[speaker] = averaged
        }

        return database
    }

    private func excludeOverlaps(in segments: [TimedSpeakerSegment], protectedSpeakers: Set<String> = []) -> [TimedSpeakerSegment] {
        guard !segments.isEmpty else { return [] }

        var sanitized: [TimedSpeakerSegment] = []

        for segment in segments {
            var adjustedStart = segment.startTimeSeconds
            let adjustedEnd = segment.endTimeSeconds

            if let previous = sanitized.last {
                if adjustedStart < previous.endTimeSeconds {
                    adjustedStart = previous.endTimeSeconds
                }
            }

            let isProtected = protectedSpeakers.contains(segment.speakerId)

            // For protected speakers, keep original timing even if fully overlapped
            // This preserves sparse speakers that overlap with dominant speakers
            if adjustedStart >= adjustedEnd {
                if isProtected {
                    // Keep the original segment without adjustment for protected speakers
                    logger.debug("Protected segment kept despite overlap: \(segment.speakerId) [\(String(format: "%.2f", segment.startTimeSeconds))-\(String(format: "%.2f", segment.endTimeSeconds))]s")
                    adjustedStart = segment.startTimeSeconds
                } else {
                    logger.debug("Segment removed (fully overlapped): \(segment.speakerId) [\(String(format: "%.2f", segment.startTimeSeconds))-\(String(format: "%.2f", segment.endTimeSeconds))]s, adjustedStart=\(String(format: "%.2f", adjustedStart))")
                    continue
                }
            }

            let duration = adjustedEnd - adjustedStart
            // Skip minSegmentDuration check for protected (rescued) speakers
            if !isProtected && duration < Float(config.minSegmentDuration) {
                continue
            }

            let originalDuration = segment.endTimeSeconds - segment.startTimeSeconds
            let qualityScale = originalDuration > 0 ? duration / originalDuration : 1
            let adjustedQuality = max(0, min(1, segment.qualityScore * qualityScale))

            let trimmed = TimedSpeakerSegment(
                speakerId: segment.speakerId,
                embedding: segment.embedding,
                startTimeSeconds: adjustedStart,
                endTimeSeconds: adjustedEnd,
                qualityScore: adjustedQuality
            )
            sanitized.append(trimmed)
        }

        return sanitized
    }

    private func appendSegment(
        cluster: Int,
        accumulator: Accumulator,
        endTime: Double,
        centroids: [[Double]],
        timedEmbeddings: [TimedEmbedding],
        output: inout [TimedSpeakerSegment]
    ) {
        guard endTime > accumulator.start else { return }
        let averageScore: Double
        if accumulator.frameCount > 0 {
            averageScore = accumulator.scoreSum / Double(accumulator.frameCount)
        } else {
            averageScore = accumulator.scoreSum
        }
        let quality = Float(min(max(averageScore, 0), 1))

        // Try to find a matching embedding for this segment's time range
        let segmentEmbedding: [Float]
        if let matchingEmbedding = findBestMatchingEmbedding(
            startTime: accumulator.start,
            endTime: endTime,
            timedEmbeddings: timedEmbeddings
        ) {
            segmentEmbedding = matchingEmbedding
        } else {
            // Fallback to centroid if no matching embedding found
            let centroidDouble =
                centroids.indices.contains(cluster)
                ? centroids[cluster]
                : Array(repeating: 0, count: centroids.first?.count ?? 0)
            segmentEmbedding = centroidDouble.map { Float($0) }
        }

        let segment = TimedSpeakerSegment(
            speakerId: "S\(cluster + 1)",
            embedding: segmentEmbedding,
            startTimeSeconds: Float(accumulator.start),
            endTimeSeconds: Float(endTime),
            qualityScore: quality
        )
        output.append(segment)
    }

    private func mergeSegments(
        _ segments: [TimedSpeakerSegment],
        gapThreshold: Double
    ) -> [TimedSpeakerSegment] {
        guard !segments.isEmpty else { return [] }

        let sorted = segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        var merged: [TimedSpeakerSegment] = []
        var current = sorted[0]

        for segment in sorted.dropFirst() {
            if segment.speakerId == current.speakerId {
                let gap = Double(segment.startTimeSeconds) - Double(current.endTimeSeconds)
                if gap <= gapThreshold {
                    let blended = blendedQuality(current, segment)
                    current = TimedSpeakerSegment(
                        speakerId: current.speakerId,
                        embedding: current.embedding,
                        startTimeSeconds: current.startTimeSeconds,
                        endTimeSeconds: max(current.endTimeSeconds, segment.endTimeSeconds),
                        qualityScore: blended
                    )
                    continue
                }
            }

            merged.append(current)
            current = segment
        }

        merged.append(current)
        return merged
    }

    private func blendedQuality(_ lhs: TimedSpeakerSegment, _ rhs: TimedSpeakerSegment) -> Float {
        let lhsDuration = Double(lhs.durationSeconds)
        let rhsDuration = Double(rhs.durationSeconds)
        let totalDuration = lhsDuration + rhsDuration

        guard totalDuration > 0 else {
            return min(max((lhs.qualityScore + rhs.qualityScore) / 2, 0), 1)
        }

        let weighted =
            Double(lhs.qualityScore) * lhsDuration
            + Double(rhs.qualityScore) * rhsDuration

        return Float(min(max(weighted / totalDuration, 0), 1))
    }

    private func sanitize(segments: [TimedSpeakerSegment]) -> [TimedSpeakerSegment] {
        var ordered = segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        let minimumDuration = max(
            Float(config.minSegmentDuration),
            Float(config.segmentationMinDurationOn)
        )

        // Count unique speakers before filtering
        let speakersBeforeFiltering = Set(ordered.map { $0.speakerId })

        // Debug: Log segment durations per speaker before filtering
        for speaker in speakersBeforeFiltering.sorted() {
            let speakerSegments = ordered.filter { $0.speakerId == speaker }
            let durations = speakerSegments.map { $0.endTimeSeconds - $0.startTimeSeconds }
            let durationsStr = durations.map { String(format: "%.2f", $0) }.joined(separator: ", ")
            logger.debug("Before filter - \(speaker): \(speakerSegments.count) segments, durations=[\(durationsStr)]s, minDuration=\(String(format: "%.2f", minimumDuration))s")
        }

        // Apply standard filtering
        var filtered = ordered.filter {
            ($0.endTimeSeconds - $0.startTimeSeconds) >= minimumDuration
        }

        // Check if filtering removed any speakers entirely
        let speakersAfterFiltering = Set(filtered.map { $0.speakerId })
        let lostSpeakers = speakersBeforeFiltering.subtracting(speakersAfterFiltering)

        // Debug: Log lost speakers
        if !lostSpeakers.isEmpty {
            logger.debug("Lost speakers after filtering: \(lostSpeakers.sorted())")
        }

        // If speakers were lost, rescue their segments with a much lower threshold
        if !lostSpeakers.isEmpty {
            // Use a very aggressive rescue threshold of 0.05s to preserve sparse speakers
            let rescueThreshold: Float = 0.05
            let rescuedSegments = ordered.filter { segment in
                lostSpeakers.contains(segment.speakerId) &&
                (segment.endTimeSeconds - segment.startTimeSeconds) >= rescueThreshold
            }

            if !rescuedSegments.isEmpty {
                logger.debug("Rescued \(rescuedSegments.count) segments for lost speakers: \(lostSpeakers)")
                filtered.append(contentsOf: rescuedSegments)
                filtered.sort { $0.startTimeSeconds < $1.startTimeSeconds }
            } else {
                // Even more aggressive: keep ALL segments for lost speakers regardless of duration
                let allLostSegments = ordered.filter { lostSpeakers.contains($0.speakerId) }
                if !allLostSegments.isEmpty {
                    logger.debug("Force-rescued \(allLostSegments.count) segments for lost speakers: \(lostSpeakers)")
                    filtered.append(contentsOf: allLostSegments)
                    filtered.sort { $0.startTimeSeconds < $1.startTimeSeconds }
                }
            }
        }

        if config.embeddingExcludeOverlap {
            // First pass: excludeOverlaps with currently known lost speakers protected
            filtered = excludeOverlaps(in: filtered, protectedSpeakers: lostSpeakers)

            // Check if excludeOverlaps removed any additional speakers
            let speakersAfterOverlaps = Set(filtered.map { $0.speakerId })
            let newlyLostSpeakers = speakersBeforeFiltering.subtracting(speakersAfterOverlaps)

            // If new speakers were lost due to overlap exclusion, rescue them
            if newlyLostSpeakers.count > lostSpeakers.count {
                let additionalLost = newlyLostSpeakers.subtracting(lostSpeakers)
                logger.debug("Speakers lost during overlap exclusion: \(additionalLost.sorted())")

                // Re-run excludeOverlaps with all lost speakers protected
                filtered = excludeOverlaps(in: ordered.filter {
                    ($0.endTimeSeconds - $0.startTimeSeconds) >= minimumDuration ||
                    newlyLostSpeakers.contains($0.speakerId)
                }, protectedSpeakers: newlyLostSpeakers)
            }
        }

        return filtered
    }

    private func chunkStartTime(
        for chunkIndex: Int,
        segmentation: SegmentationOutput
    ) -> Double {
        if segmentation.chunkOffsets.indices.contains(chunkIndex) {
            return segmentation.chunkOffsets[chunkIndex]
        } else {
            return Double(chunkIndex) * config.windowDuration
        }
    }

    /// Find the best matching embedding for a segment by time overlap.
    /// Returns the embedding with the highest temporal overlap with the segment.
    private func findBestMatchingEmbedding(
        startTime: Double,
        endTime: Double,
        timedEmbeddings: [TimedEmbedding]
    ) -> [Float]? {
        guard !timedEmbeddings.isEmpty else { return nil }

        var bestEmbedding: [Float]?
        var bestOverlap: Double = 0

        for embedding in timedEmbeddings {
            let overlapStart = max(startTime, embedding.startTime)
            let overlapEnd = min(endTime, embedding.endTime)
            let overlap = overlapEnd - overlapStart

            if overlap > bestOverlap {
                bestOverlap = overlap
                bestEmbedding = embedding.embedding256
            }
        }

        return bestEmbedding
    }
}
