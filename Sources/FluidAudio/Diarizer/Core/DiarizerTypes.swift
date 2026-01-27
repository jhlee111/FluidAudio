import Foundation

// MARK: - Diarization Pipeline Types
// This file contains types used in the diarization processing pipeline
// For speaker profile types, see SpeakerTypes.swift

public struct DiarizerConfig: Sendable {
    /// Threshold for clustering speaker embeddings (0.5-0.9). Lower = more speakers.
    public var clusteringThreshold: Float = 0.7

    /// Minimum speech segment duration in seconds. Shorter segments are discarded.
    public var minSpeechDuration: Float = 1.0

    /// Minimum segment duration for updating speaker embeddings (seconds).
    public var minEmbeddingUpdateDuration: Float = 2.0

    /// Minimum silence gap (seconds) before splitting same speaker's segments.
    public var minSilenceGap: Float = 0.5

    /// Expected number of speakers (-1 for automatic).
    public var numClusters: Int = -1

    /// Minimum active frames for valid speech detection.
    public var minActiveFramesCount: Float = 10.0

    /// Enable debug logging.
    public var debugMode: Bool = false

    /// Duration of audio chunks for processing (seconds).
    public var chunkDuration: Float = 10.0

    /// Overlap between chunks (seconds).
    public var chunkOverlap: Float = 0.0

    public static let `default` = DiarizerConfig()

    public init(
        clusteringThreshold: Float = 0.7,
        minSpeechDuration: Float = 1.0,
        minEmbeddingUpdateDuration: Float = 2.0,
        minSilenceGap: Float = 0.5,
        numClusters: Int = -1,
        minActiveFramesCount: Float = 10.0,
        debugMode: Bool = false,
        chunkDuration: Float = 10.0,
        chunkOverlap: Float = 0.0
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.minSpeechDuration = minSpeechDuration
        self.minEmbeddingUpdateDuration = minEmbeddingUpdateDuration
        self.minSilenceGap = minSilenceGap
        self.numClusters = numClusters
        self.minActiveFramesCount = minActiveFramesCount
        self.debugMode = debugMode
        self.chunkDuration = chunkDuration
        self.chunkOverlap = chunkOverlap
    }
}

public struct PipelineTimings: Sendable, Codable {
    public let modelCompilationSeconds: TimeInterval
    public let audioLoadingSeconds: TimeInterval
    public let segmentationSeconds: TimeInterval
    public let embeddingExtractionSeconds: TimeInterval
    public let speakerClusteringSeconds: TimeInterval
    public let postProcessingSeconds: TimeInterval
    public let totalInferenceSeconds: TimeInterval
    public let totalProcessingSeconds: TimeInterval

    public init(
        modelCompilationSeconds: TimeInterval = 0,
        audioLoadingSeconds: TimeInterval = 0,
        segmentationSeconds: TimeInterval = 0,
        embeddingExtractionSeconds: TimeInterval = 0,
        speakerClusteringSeconds: TimeInterval = 0,
        postProcessingSeconds: TimeInterval = 0
    ) {
        self.modelCompilationSeconds = modelCompilationSeconds
        self.audioLoadingSeconds = audioLoadingSeconds
        self.segmentationSeconds = segmentationSeconds
        self.embeddingExtractionSeconds = embeddingExtractionSeconds
        self.speakerClusteringSeconds = speakerClusteringSeconds
        self.postProcessingSeconds = postProcessingSeconds
        self.totalInferenceSeconds =
            segmentationSeconds + embeddingExtractionSeconds + speakerClusteringSeconds
        self.totalProcessingSeconds =
            modelCompilationSeconds + audioLoadingSeconds
            + segmentationSeconds + embeddingExtractionSeconds + speakerClusteringSeconds
            + postProcessingSeconds
    }

    public var stagePercentages: [String: Double] {
        guard totalProcessingSeconds > 0 else {
            return [:]
        }

        return [
            "Model Compilation": (modelCompilationSeconds / totalProcessingSeconds) * 100,
            "Audio Loading": (audioLoadingSeconds / totalProcessingSeconds) * 100,
            "Segmentation": (segmentationSeconds / totalProcessingSeconds) * 100,
            "Embedding Extraction": (embeddingExtractionSeconds / totalProcessingSeconds) * 100,
            "Speaker Clustering": (speakerClusteringSeconds / totalProcessingSeconds) * 100,
            "Post Processing": (postProcessingSeconds / totalProcessingSeconds) * 100,
        ]
    }

    public var bottleneckStage: String {
        let stages = [
            ("Model Compilation", modelCompilationSeconds),
            ("Audio Loading", audioLoadingSeconds),
            ("Segmentation", segmentationSeconds),
            ("Embedding Extraction", embeddingExtractionSeconds),
            ("Speaker Clustering", speakerClusteringSeconds),
            ("Post Processing", postProcessingSeconds),
        ]

        return stages.max(by: { $0.1 < $1.1 })?.0 ?? "Unknown"
    }
}

/// Powerset class labels for the segmentation model output
/// Index corresponds to the combination of active speakers
public enum PowersetClass: Int, CaseIterable, Sendable {
    case silence = 0      // No one speaking
    case speaker1 = 1     // Speaker 1 only (S1)
    case speaker2 = 2     // Speaker 2 only (S2)
    case speaker3 = 3     // Speaker 3 only (S3)
    case overlap12 = 4    // S1+S2 overlapping
    case overlap13 = 5    // S1+S3 overlapping
    case overlap23 = 6    // S2+S3 overlapping
    case overlapAll = 7   // All 3 overlapping

    public var label: String {
        switch self {
        case .silence: return "silence"
        case .speaker1: return "S1"
        case .speaker2: return "S2"
        case .speaker3: return "S3"
        case .overlap12: return "S1+S2"
        case .overlap13: return "S1+S3"
        case .overlap23: return "S2+S3"
        case .overlapAll: return "S1+S2+S3"
        }
    }

    public var isOverlap: Bool {
        self == .overlap12 || self == .overlap13 || self == .overlap23 || self == .overlapAll
    }
}

/// Frame-level analysis data for advanced processing pipelines
public struct FrameLevelAnalysis: Sendable {
    /// Number of active speakers per frame (1 = single speaker, 2+ = overlap)
    public let speakerCount: [Int]

    /// Voice activity confidence per frame (0.0-1.0)
    public let voiceActivityConfidence: [Float]

    /// Raw speaker weights per frame: [globalFrame][speakerIndex] -> probability (0.0-1.0)
    /// Each inner array contains probabilities for each detected speaker at that frame
    /// This is the low-level data from the segmentation model before thresholding
    public let speakerWeights: [[Float]]

    /// Raw powerset class probabilities per frame: [globalFrame][classIndex] -> probability
    /// 8 classes: silence, S1, S2, S3, S1+S2, S1+S3, S2+S3, S1+S2+S3
    /// Use PowersetClass enum to interpret indices
    public let powersetProbs: [[Float]]

    /// Frame duration in seconds
    public let frameDuration: Double

    /// Total duration of the analyzed audio in seconds
    public let totalDuration: Double

    /// Number of speakers detected by the segmentation model
    public let numSpeakers: Int

    public init(
        speakerCount: [Int],
        voiceActivityConfidence: [Float],
        speakerWeights: [[Float]] = [],
        powersetProbs: [[Float]] = [],
        frameDuration: Double,
        totalDuration: Double,
        numSpeakers: Int = 0
    ) {
        self.speakerCount = speakerCount
        self.voiceActivityConfidence = voiceActivityConfidence
        self.speakerWeights = speakerWeights
        self.powersetProbs = powersetProbs
        self.frameDuration = frameDuration
        self.totalDuration = totalDuration
        self.numSpeakers = numSpeakers
    }

    /// Get speaker count at a specific time
    public func speakerCount(at time: Double) -> Int {
        guard frameDuration > 0, !speakerCount.isEmpty else { return 0 }
        let frameIndex = min(Int(time / frameDuration), speakerCount.count - 1)
        return frameIndex >= 0 ? speakerCount[frameIndex] : 0
    }

    /// Get voice activity confidence at a specific time
    public func voiceActivity(at time: Double) -> Float {
        guard frameDuration > 0, !voiceActivityConfidence.isEmpty else { return 0 }
        let frameIndex = min(Int(time / frameDuration), voiceActivityConfidence.count - 1)
        return frameIndex >= 0 ? voiceActivityConfidence[frameIndex] : 0
    }

    /// Get raw speaker weights at a specific time
    /// Returns array of probabilities for each speaker (0.0-1.0)
    public func speakerWeights(at time: Double) -> [Float] {
        guard frameDuration > 0, !speakerWeights.isEmpty else { return [] }
        let frameIndex = min(Int(time / frameDuration), speakerWeights.count - 1)
        return frameIndex >= 0 ? speakerWeights[frameIndex] : []
    }

    /// Get speaker count for a time range (returns counts for all frames in range)
    public func speakerCounts(in range: ClosedRange<Double>) -> [Int] {
        guard frameDuration > 0, !speakerCount.isEmpty else { return [] }
        let startFrame = max(0, Int(range.lowerBound / frameDuration))
        let endFrame = min(speakerCount.count - 1, Int(range.upperBound / frameDuration))
        guard startFrame <= endFrame else { return [] }
        return Array(speakerCount[startFrame...endFrame])
    }

    /// Get raw speaker weights for a time range
    /// Returns array of per-frame speaker weight arrays
    public func speakerWeights(in range: ClosedRange<Double>) -> [[Float]] {
        guard frameDuration > 0, !speakerWeights.isEmpty else { return [] }
        let startFrame = max(0, Int(range.lowerBound / frameDuration))
        let endFrame = min(speakerWeights.count - 1, Int(range.upperBound / frameDuration))
        guard startFrame <= endFrame else { return [] }
        return Array(speakerWeights[startFrame...endFrame])
    }

    /// Get detailed frame data for a time range (for debugging/analysis)
    /// Returns array of (time, speakerCount, voiceActivity, speakerWeights) tuples
    public func frameData(in range: ClosedRange<Double>) -> [(time: Double, speakerCount: Int, voiceActivity: Float, weights: [Float])] {
        guard frameDuration > 0 else { return [] }
        let startFrame = max(0, Int(range.lowerBound / frameDuration))
        let endFrame = min(speakerCount.count - 1, Int(range.upperBound / frameDuration))
        guard startFrame <= endFrame else { return [] }

        var result: [(time: Double, speakerCount: Int, voiceActivity: Float, weights: [Float])] = []
        for frame in startFrame...endFrame {
            let time = Double(frame) * frameDuration
            let count = speakerCount[frame]
            let vad = frame < voiceActivityConfidence.count ? voiceActivityConfidence[frame] : 0
            let weights = frame < speakerWeights.count ? speakerWeights[frame] : []
            result.append((time: time, speakerCount: count, voiceActivity: vad, weights: weights))
        }
        return result
    }
}

public struct DiarizationResult: Sendable {
    public let segments: [TimedSpeakerSegment]

    /// Speaker database with embeddings (populated by offline pipelines for downstream use)
    public let speakerDatabase: [String: [Float]]?

    /// Performance timings collected during diarization
    public let timings: PipelineTimings?

    /// Frame-level analysis data (speaker counts, voice activity per frame)
    public let frameLevelAnalysis: FrameLevelAnalysis?

    public init(
        segments: [TimedSpeakerSegment],
        speakerDatabase: [String: [Float]]? = nil,
        timings: PipelineTimings? = nil,
        frameLevelAnalysis: FrameLevelAnalysis? = nil
    ) {
        self.segments = segments
        self.speakerDatabase = speakerDatabase
        self.timings = timings
        self.frameLevelAnalysis = frameLevelAnalysis
    }
}

/// Represents a segment of speech from a specific speaker with timing information
/// This is used for diarization results - "who spoke when"
/// For speaker profiles, see Speaker struct in SpeakerTypes.swift
public struct TimedSpeakerSegment: Sendable, Identifiable {
    public let id = UUID()
    public let speakerId: String
    public let embedding: [Float]
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let qualityScore: Float

    // MARK: - Overlap Information

    /// Ratio of frames in this segment where multiple speakers are active (0.0-1.0)
    public let overlapRatio: Float

    /// True if overlapRatio exceeds the overlap threshold (default 0.1)
    public var isOverlapping: Bool {
        overlapRatio > 0.1
    }

    /// Speaker IDs that overlap with this segment (excluding self)
    public let overlappingSpeakers: [String]

    // MARK: - Computed Properties

    public var durationSeconds: Float {
        endTimeSeconds - startTimeSeconds
    }

    /// True if this segment is clean (single speaker, no overlap)
    public var isClean: Bool {
        !isOverlapping && overlappingSpeakers.isEmpty
    }

    // MARK: - Initializers

    public init(
        speakerId: String,
        embedding: [Float],
        startTimeSeconds: Float,
        endTimeSeconds: Float,
        qualityScore: Float,
        overlapRatio: Float = 0.0,
        overlappingSpeakers: [String] = []
    ) {
        self.speakerId = speakerId
        self.embedding = embedding
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.qualityScore = qualityScore
        self.overlapRatio = overlapRatio
        self.overlappingSpeakers = overlappingSpeakers
    }
}

public struct ModelPaths: Sendable {
    public let segmentationPath: URL
    public let embeddingPath: URL

    public init(segmentationPath: URL, embeddingPath: URL) {
        self.segmentationPath = segmentationPath
        self.embeddingPath = embeddingPath
    }
}

public struct AudioValidationResult: Sendable {
    public let isValid: Bool
    public let durationSeconds: Float
    public let issues: [String]

    public init(isValid: Bool, durationSeconds: Float, issues: [String] = []) {
        self.isValid = isValid
        self.durationSeconds = durationSeconds
        self.issues = issues
    }
}

public enum DiarizerError: Error, LocalizedError {
    case notInitialized
    case modelDownloadFailed
    case modelCompilationFailed
    case embeddingExtractionFailed
    case invalidAudioData
    case processingFailed(String)
    case memoryAllocationFailed
    case invalidArrayBounds

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarization system not initialized. Call initialize() first."
        case .modelDownloadFailed:
            return "Failed to download required models."
        case .modelCompilationFailed:
            return "Failed to compile CoreML models."
        case .embeddingExtractionFailed:
            return "Failed to extract speaker embedding from audio."
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .memoryAllocationFailed:
            return "Failed to allocate ANE-aligned memory."
        case .invalidArrayBounds:
            return "Array bounds exceeded for zero-copy view."
        }
    }
}
