#!/usr/bin/env python3
"""
Model evaluation script for Voice AI Agent project.
Tests different Whisper models for accuracy, latency, and resource usage.
"""

import os
import sys
import time
import argparse
import asyncio
import logging
import numpy as np
import psutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from speech_to_text.streaming.whisper_streaming import StreamingWhisperASR
    from speech_to_text.utils.audio_utils import load_audio_file
except ImportError:
    print("Could not import speech_to_text modules. Make sure you're in the correct directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define available models to test
AVAILABLE_MODELS = [
    "tiny.en", "tiny.en-q5_1", 
    "base.en", "base.en-q5_1",
    "small.en", "small.en-q5_1",
    "medium.en", "medium.en-q5_0",
    "large-v3", "large-v3-q5_0"
]

# Define typical customer service evaluation parameters
SERVICE_PARAMETERS = {
    "default": {
        "temperature": 0.0,
        "initial_prompt": None,
        "max_tokens": 0,
        "no_context": False,
        "single_segment": True
    },
    "customer_service": {
        "temperature": 0.2,
        "initial_prompt": "Customer service conversation transcript:",
        "max_tokens": 150,
        "no_context": False,
        "single_segment": True
    },
    "technical_support": {
        "temperature": 0.1,
        "initial_prompt": "Technical support call transcript with technical terminology:",
        "max_tokens": 200,
        "no_context": False,
        "single_segment": True
    }
}

class ModelEvaluator:
    """Evaluates different Whisper models for the Voice AI Agent project."""
    
    def __init__(self, 
                 test_audio_dir: str,
                 reference_transcripts_dir: Optional[str] = None,
                 output_dir: str = "model_evaluation_results",
                 parameter_preset: str = "customer_service",
                 n_threads: int = 4):
        """
        Initialize the model evaluator.
        
        Args:
            test_audio_dir: Directory containing test audio files
            reference_transcripts_dir: Directory containing reference transcripts
            output_dir: Directory to save evaluation results
            parameter_preset: Parameter preset to use for evaluation
            n_threads: Number of CPU threads to use
        """
        self.test_audio_dir = Path(test_audio_dir)
        self.reference_transcripts_dir = Path(reference_transcripts_dir) if reference_transcripts_dir else None
        self.output_dir = Path(output_dir)
        self.n_threads = n_threads
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set parameters based on preset
        if parameter_preset in SERVICE_PARAMETERS:
            self.parameters = SERVICE_PARAMETERS[parameter_preset]
            logger.info(f"Using parameter preset: {parameter_preset}")
        else:
            self.parameters = SERVICE_PARAMETERS["default"]
            logger.warning(f"Preset {parameter_preset} not found, using default")
            
        # Find test audio files
        self.test_files = []
        for ext in [".wav", ".mp3", ".m4a", ".flac"]:
            self.test_files.extend(list(self.test_audio_dir.glob(f"*{ext}")))
            
        if not self.test_files:
            logger.error(f"No audio files found in {test_audio_dir}")
            sys.exit(1)
            
        logger.info(f"Found {len(self.test_files)} test audio files")
        
        # Load reference transcripts if available
        self.reference_transcripts = {}
        if self.reference_transcripts_dir:
            for ref_file in self.reference_transcripts_dir.glob("*.txt"):
                # Get the base filename without extension to match with audio files
                base_name = ref_file.stem
                with open(ref_file, "r") as f:
                    self.reference_transcripts[base_name] = f.read().strip()
                    
            logger.info(f"Loaded {len(self.reference_transcripts)} reference transcripts")
    
    async def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model on all test files.
        
        Args:
            model_name: Name of the model to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        results = {
            "model": model_name,
            "parameters": self.parameters,
            "overall_latency_ms": 0,
            "memory_usage_mb": 0,
            "audio_duration_seconds": 0,
            "realtime_factor": 0,
            "files": []
        }
        
        # Create ASR instance
        try:
            # Track memory before model creation
            memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            # Create ASR instance
            start_time = time.time()
            asr = StreamingWhisperASR(
                model_path=model_name,
                n_threads=self.n_threads,
                **self.parameters
            )
            model_load_time = time.time() - start_time
            
            # Track memory after model creation
            memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_usage = memory_after - memory_before
            
            results["model_load_time_ms"] = model_load_time * 1000
            results["memory_usage_mb"] = memory_usage
            
            logger.info(f"Model loaded in {model_load_time:.2f}s, memory usage: {memory_usage:.2f} MB")
            
            # Process each test file
            total_audio_duration = 0
            total_processing_time = 0
            
            for audio_file in self.test_files:
                file_result = await self._process_file(asr, audio_file)
                results["files"].append(file_result)
                
                total_audio_duration += file_result["audio_duration_seconds"]
                total_processing_time += file_result["processing_time_seconds"]
            
            # Calculate overall metrics
            results["audio_duration_seconds"] = total_audio_duration
            results["overall_latency_ms"] = (total_processing_time / len(self.test_files)) * 1000
            results["realtime_factor"] = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
            
            logger.info(f"Model evaluation complete:")
            logger.info(f"  Average latency: {results['overall_latency_ms']:.2f} ms")
            logger.info(f"  Realtime factor: {results['realtime_factor']:.2f}x")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            results["error"] = str(e)
            return results
    
    async def _process_file(self, asr: StreamingWhisperASR, audio_file: Path) -> Dict[str, Any]:
        """
        Process a single audio file with the given ASR instance.
        
        Args:
            asr: StreamingWhisperASR instance
            audio_file: Path to audio file
            
        Returns:
            Dictionary of file processing results
        """
        file_name = audio_file.name
        logger.info(f"Processing file: {file_name}")
        
        result = {
            "file_name": file_name,
            "processing_time_seconds": 0,
            "audio_duration_seconds": 0,
            "transcript": "",
            "wer": None
        }
        
        try:
            # Load audio file
            audio, sample_rate = load_audio_file(str(audio_file), target_sr=asr.sample_rate)
            
            # Calculate audio duration
            audio_duration = len(audio) / sample_rate
            result["audio_duration_seconds"] = audio_duration
            
            # Process audio
            start_time = time.time()
            
            # Start streaming
            asr.start_streaming()
            
            # Process the entire audio as one chunk
            await asr.process_audio_chunk(audio)
            
            # Get final transcript
            transcript, _ = await asr.stop_streaming()
            result["transcript"] = transcript
            
            processing_time = time.time() - start_time
            result["processing_time_seconds"] = processing_time
            
            # Calculate realtime factor
            result["realtime_factor"] = processing_time / audio_duration if audio_duration > 0 else 0
            
            logger.info(f"  Audio duration: {audio_duration:.2f}s")
            logger.info(f"  Processing time: {processing_time:.2f}s")
            logger.info(f"  Realtime factor: {result['realtime_factor']:.2f}x")
            
            # Calculate WER if reference transcript is available
            base_name = audio_file.stem
            if base_name in self.reference_transcripts:
                wer = self._calculate_wer(transcript, self.reference_transcripts[base_name])
                result["wer"] = wer
                logger.info(f"  WER: {wer:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            result["error"] = str(e)
            return result
    
    def _calculate_wer(self, hypothesis: str, reference: str) -> float:
        """
        Calculate Word Error Rate (WER) between hypothesis and reference.
        
        Args:
            hypothesis: Transcribed text
            reference: Reference transcript
            
        Returns:
            WER as a percentage
        """
        # Normalize texts
        def normalize_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return text
        
        hyp = normalize_text(hypothesis).split()
        ref = normalize_text(reference).split()
        
        # Calculate Levenshtein distance
        d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.uint32)
        
        # Initialize first column and row
        for i in range(len(ref) + 1):
            d[i, 0] = i
        for j in range(len(hyp) + 1):
            d[0, j] = j
            
        # Fill the matrix
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i-1] == hyp[j-1]:
                    d[i, j] = d[i-1, j-1]
                else:
                    substitution = d[i-1, j-1] + 1
                    insertion = d[i, j-1] + 1
                    deletion = d[i-1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)
        
        # Calculate WER
        wer = float(d[len(ref), len(hyp)]) / len(ref) * 100
        return wer
    
    async def evaluate_all_models(self, models_to_test: Optional[List[str]] = None) -> None:
        """
        Evaluate all specified models and save results.
        
        Args:
            models_to_test: List of models to test, or None to test all
        """
        if not models_to_test:
            models_to_test = AVAILABLE_MODELS
            
        all_results = []
        
        for model in models_to_test:
            try:
                model_results = await self.evaluate_model(model)
                all_results.append(model_results)
                
                # Save individual model results
                with open(self.output_dir / f"{model}_results.json", "w") as f:
                    json.dump(model_results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Failed to evaluate model {model}: {e}")
        
        # Save summary results
        summary = self._create_summary(all_results)
        with open(self.output_dir / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        # Generate readable report
        self._generate_report(summary)
        
        logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
    
    def _create_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of all model results.
        
        Args:
            all_results: List of individual model results
            
        Returns:
            Dictionary containing summary metrics
        """
        summary = {
            "models": [],
            "best_latency_model": None,
            "best_accuracy_model": None,
            "best_overall_model": None,
            "evaluation_parameters": self.parameters
        }
        
        if not all_results:
            return summary
            
        # Extract key metrics for each model
        best_latency = float('inf')
        best_accuracy = float('inf')  # Lower WER is better
        best_overall_score = float('inf')  # Combined metric
        
        for result in all_results:
            model_name = result["model"]
            latency = result.get("overall_latency_ms", float('inf'))
            
            # Calculate average WER across files
            total_wer = 0
            wer_count = 0
            for file_result in result.get("files", []):
                if file_result.get("wer") is not None:
                    total_wer += file_result["wer"]
                    wer_count += 1
                    
            avg_wer = total_wer / wer_count if wer_count > 0 else None
            
            # Calculate memory per second of audio
            memory_per_second = result.get("memory_usage_mb", 0) / result.get("audio_duration_seconds", 1)
            
            # Calculate overall score (weighted combination of latency and accuracy)
            # For our voice agent where latency is critical, weight it more heavily
            # WER weight: 40%, Latency weight: 60%
            if avg_wer is not None:
                overall_score = (0.4 * avg_wer) + (0.6 * (latency / 100))  # Normalize latency
            else:
                overall_score = latency / 100  # Just use latency if no WER
                
            model_summary = {
                "model": model_name,
                "latency_ms": latency,
                "realtime_factor": result.get("realtime_factor", None),
                "memory_usage_mb": result.get("memory_usage_mb", None),
                "memory_per_second_mb": memory_per_second,
                "avg_wer": avg_wer,
                "overall_score": overall_score
            }
            
            summary["models"].append(model_summary)
            
            # Update best models
            if latency < best_latency:
                best_latency = latency
                summary["best_latency_model"] = model_name
                
            if avg_wer is not None and avg_wer < best_accuracy:
                best_accuracy = avg_wer
                summary["best_accuracy_model"] = model_name
                
            if overall_score < best_overall_score:
                best_overall_score = overall_score
                summary["best_overall_model"] = model_name
                
        # Sort models by overall score
        summary["models"].sort(key=lambda x: x.get("overall_score", float('inf')))
        
        return summary
    
    def _generate_report(self, summary: Dict[str, Any]) -> None:
        """
        Generate a readable report from the summary.
        
        Args:
            summary: Evaluation summary dictionary
        """
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, "w") as f:
            f.write("=== Whisper Model Evaluation Report ===\n\n")
            f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of test files: {len(self.test_files)}\n")
            f.write(f"Number of models evaluated: {len(summary['models'])}\n\n")
            
            f.write("--- Parameter Settings ---\n")
            for key, value in summary["evaluation_parameters"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("--- Best Models ---\n")
            f.write(f"Best Overall Model: {summary['best_overall_model']}\n")
            f.write(f"Best Latency Model: {summary['best_latency_model']}\n")
            f.write(f"Best Accuracy Model: {summary['best_accuracy_model']}\n\n")
            
            f.write("--- Model Rankings (by overall score) ---\n")
            for i, model in enumerate(summary["models"], 1):
                f.write(f"{i}. {model['model']}:\n")
                f.write(f"   - Latency: {model.get('latency_ms', 'N/A'):.2f} ms\n")
                f.write(f"   - Realtime Factor: {model.get('realtime_factor', 'N/A'):.2f}x\n")
                f.write(f"   - Memory Usage: {model.get('memory_usage_mb', 'N/A'):.2f} MB\n")
                f.write(f"   - Avg WER: {model.get('avg_wer', 'N/A'):.2f}%\n")
                f.write(f"   - Overall Score: {model.get('overall_score', 'N/A'):.4f}\n\n")
            
            f.write("--- Recommendations for Voice AI Agent ---\n")
            f.write("Based on the requirement of <500ms STT latency and the need for high accuracy\n")
            f.write("in a customer service context, the recommended model configuration is:\n\n")
            
            # Find the best model that meets latency requirement
            suitable_models = [m for m in summary["models"] if m.get("latency_ms", float('inf')) < 500]
            if suitable_models:
                best_suitable = suitable_models[0]
                f.write(f"Recommended Model: {best_suitable['model']}\n")
                f.write(f"Expected Latency: {best_suitable.get('latency_ms', 'N/A'):.2f} ms\n")
                f.write(f"Expected Accuracy (WER): {best_suitable.get('avg_wer', 'N/A'):.2f}%\n")
                f.write(f"Memory Requirement: {best_suitable.get('memory_usage_mb', 'N/A'):.2f} MB\n")
            else:
                f.write("No model meets the 500ms latency requirement. Consider:\n")
                f.write("1. Using a quantized model with some accuracy trade-off\n")
                f.write("2. Reducing audio chunk size to improve streaming performance\n")
                f.write("3. Allocating more CPU threads or using GPU acceleration\n")
            
            f.write("\n--- Parameter Recommendations ---\n")
            f.write("Based on the customer service use case, the following parameter settings are recommended:\n\n")
            f.write("temperature: 0.2  # Balanced between accuracy and adaptability\n")
            f.write("initial_prompt: \"Customer service conversation transcript:\"\n")
            f.write("max_tokens: 150   # Appropriate for typical customer utterances\n")
            f.write("no_context: False # Use context for better continuous recognition\n")
            f.write("single_segment: True # Better for real-time applications\n")
            
        logger.info(f"Evaluation report saved to {report_path}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Whisper Model Evaluation for Voice AI Agent')
    parser.add_argument('--audio-dir', type=str, required=True,
                      help='Directory containing test audio files')
    parser.add_argument('--transcript-dir', type=str, default=None,
                      help='Directory containing reference transcripts (optional)')
    parser.add_argument('--output-dir', type=str, default='model_evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--models', type=str, default=None,
                      help='Comma-separated list of models to test (default: all)')
    parser.add_argument('--threads', type=int, default=4,
                      help='Number of CPU threads to use')
    parser.add_argument('--preset', type=str, default='customer_service',
                      choices=list(SERVICE_PARAMETERS.keys()),
                      help='Parameter preset to use')
    
    args = parser.parse_args()
    
    # Parse models list if provided
    models_to_test = None
    if args.models:
        models_to_test = [model.strip() for model in args.models.split(",")]
        for model in models_to_test:
            if model not in AVAILABLE_MODELS:
                logger.warning(f"Model {model} is not in the list of available models.")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        test_audio_dir=args.audio_dir,
        reference_transcripts_dir=args.transcript_dir,
        output_dir=args.output_dir,
        parameter_preset=args.preset,
        n_threads=args.threads
    )
    
    # Evaluate models
    await evaluator.evaluate_all_models(models_to_test)
    
    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)