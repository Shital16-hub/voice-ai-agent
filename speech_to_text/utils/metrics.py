"""
Metrics utilities for the speech-to-text module.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    """Latency metrics for speech recognition."""
    chunk_processing_time: float  # Time to process a single chunk (ms)
    model_inference_time: float   # Time spent in model inference (ms)
    total_latency: float          # Total latency from audio to transcription (ms)
    audio_duration: float         # Duration of audio processed (ms)
    real_time_factor: float       # Ratio of processing time to audio duration

class MetricsCollector:
    """
    Metrics collector for speech recognition.
    
    This class collects and reports metrics about the speech recognition process,
    including latency, accuracy, and throughput.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize MetricsCollector.
        
        Args:
            enabled: Whether to enable metrics collection
        """
        self.enabled = enabled
        
        # Latency metrics
        self.chunk_processing_times: List[float] = []
        self.model_inference_times: List[float] = []
        self.total_latencies: List[float] = []
        self.audio_durations: List[float] = []
        
        # Performance metrics
        self.processed_chunks = 0
        self.processed_audio_duration = 0.0
        self.start_time = time.time()
        
        # Prometheus metrics
        self._setup_prometheus() if enabled and PROMETHEUS_AVAILABLE else None
        
        logger.info(f"Metrics collection {'enabled' if enabled else 'disabled'}")
    
    def _setup_prometheus(self):
        """Set up Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics not exposed")
            return
        
        # Create metrics
        self.prom_chunk_processing_time = prom.Histogram(
            "stt_chunk_processing_time_seconds",
            "Time to process a single audio chunk",
            buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]
        )
        
        self.prom_model_inference_time = prom.Histogram(
            "stt_model_inference_time_seconds",
            "Time spent in model inference",
            buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]
        )
        
        self.prom_total_latency = prom.Histogram(
            "stt_total_latency_seconds",
            "Total latency from audio to transcription",
            buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]
        )
        
        self.prom_real_time_factor = prom.Gauge(
            "stt_real_time_factor",
            "Ratio of processing time to audio duration"
        )
        
        self.prom_processed_chunks = prom.Counter(
            "stt_processed_chunks_total",
            "Total number of audio chunks processed"
        )
        
        self.prom_processed_audio = prom.Counter(
            "stt_processed_audio_seconds_total",
            "Total duration of audio processed in seconds"
        )
        
        # Start Prometheus HTTP server in a separate thread
        try:
            threading.Thread(
                target=prom.start_http_server,
                args=(8000,),
                daemon=True
            ).start()
            logger.info("Started Prometheus metrics server on port 8000")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
    
    def record_chunk_processing(
        self,
        chunk_processing_time: float,
        model_inference_time: float,
        audio_duration: float
    ):
        """
        Record metrics for a single chunk processing.
        
        Args:
            chunk_processing_time: Time to process the chunk in seconds
            model_inference_time: Time spent in model inference in seconds
            audio_duration: Duration of audio in the chunk in seconds
        """
        if not self.enabled:
            return
        
        # Convert to milliseconds for internal storage
        chunk_processing_time_ms = chunk_processing_time * 1000
        model_inference_time_ms = model_inference_time * 1000
        audio_duration_ms = audio_duration * 1000
        
        # Calculate total latency
        total_latency_ms = chunk_processing_time_ms
        
        # Record metrics
        self.chunk_processing_times.append(chunk_processing_time_ms)
        self.model_inference_times.append(model_inference_time_ms)
        self.total_latencies.append(total_latency_ms)
        self.audio_durations.append(audio_duration_ms)
        
        # Update counters
        self.processed_chunks += 1
        self.processed_audio_duration += audio_duration
        
        # Calculate real-time factor
        real_time_factor = chunk_processing_time / audio_duration if audio_duration > 0 else 0
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.prom_chunk_processing_time.observe(chunk_processing_time)
            self.prom_model_inference_time.observe(model_inference_time)
            self.prom_total_latency.observe(total_latency_ms / 1000)  # Convert to seconds
            self.prom_real_time_factor.set(real_time_factor)
            self.prom_processed_chunks.inc()
            self.prom_processed_audio.inc(audio_duration)
        
        # Log metrics
        if self.processed_chunks % 100 == 0:
            self.log_summary()
    
    def get_latency_metrics(self) -> LatencyMetrics:
        """
        Get current latency metrics.
        
        Returns:
            LatencyMetrics object with current metrics
        """
        if not self.chunk_processing_times:
            return LatencyMetrics(
                chunk_processing_time=0,
                model_inference_time=0,
                total_latency=0,
                audio_duration=0,
                real_time_factor=0
            )
        
        # Calculate averages
        avg_chunk_processing_time = sum(self.chunk_processing_times) / len(self.chunk_processing_times)
        avg_model_inference_time = sum(self.model_inference_times) / len(self.model_inference_times)
        avg_total_latency = sum(self.total_latencies) / len(self.total_latencies)
        avg_audio_duration = sum(self.audio_durations) / len(self.audio_durations)
        
        # Calculate real-time factor
        real_time_factor = avg_chunk_processing_time / avg_audio_duration if avg_audio_duration > 0 else 0
        
        return LatencyMetrics(
            chunk_processing_time=avg_chunk_processing_time,
            model_inference_time=avg_model_inference_time,
            total_latency=avg_total_latency,
            audio_duration=avg_audio_duration,
            real_time_factor=real_time_factor
        )
    
    def log_summary(self):
        """Log a summary of current metrics."""
        if not self.enabled:
            return
        
        metrics = self.get_latency_metrics()
        
        # Calculate throughput
        elapsed_time = time.time() - self.start_time
        throughput = self.processed_audio_duration / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(
            f"Metrics summary: processed {self.processed_chunks} chunks, "
            f"{self.processed_audio_duration:.1f}s of audio in {elapsed_time:.1f}s "
            f"(throughput: {throughput:.2f}x real-time)"
        )
        
        logger.info(
            f"Latency metrics: chunk={metrics.chunk_processing_time:.1f}ms, "
            f"model={metrics.model_inference_time:.1f}ms, "
            f"total={metrics.total_latency:.1f}ms, "
            f"RTF={metrics.real_time_factor:.2f}"
        )
    
    def reset(self):
        """Reset all metrics."""
        self.chunk_processing_times = []
        self.model_inference_times = []
        self.total_latencies = []
        self.audio_durations = []
        self.processed_chunks = 0
        self.processed_audio_duration = 0.0
        self.start_time = time.time()