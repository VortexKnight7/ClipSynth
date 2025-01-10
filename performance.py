import time
import torch
import psutil
import os
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from speech_recognition import Recognizer, AudioFile
from rouge import Rouge
from difflib import SequenceMatcher
import json

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "execution_time": {},
            "model_latency": None,
            "transcription_accuracy": None,
            "memory_usage": None,
            "user_response_time": None,
            "error_rate": 0,
            "rouge_scores": {},
        }
        self.error_count = 0
        self.total_requests = 0

    def record_execution_time(self, stage, start_time, end_time):
        self.metrics["execution_time"][stage] = end_time - start_time
        print(f"[DEBUG] Execution time for {stage}: {end_time - start_time} seconds")

    def record_model_latency(self, latency):
        self.metrics["model_latency"] = latency
        print(f"[DEBUG] Model latency: {latency} seconds")

    def record_transcription_accuracy(self, reference, hypothesis):
        def word_error_rate(r, h):
            r_words = r.split()
            h_words = h.split()
            distance = SequenceMatcher(None, r_words, h_words).ratio()
            return 1 - distance

        self.metrics["transcription_accuracy"] = word_error_rate(reference, hypothesis)
        print(f"[DEBUG] Transcription accuracy (WER): {self.metrics['transcription_accuracy']}")

    def record_memory_usage(self):
        self.metrics["memory_usage"] = psutil.virtual_memory().used / (1024 * 1024)  # Memory in MB
        print(f"[DEBUG] Memory usage: {self.metrics['memory_usage']} MB")

    def increment_error_count(self):
        self.error_count += 1
        print(f"[DEBUG] Error count incremented. Total errors: {self.error_count}")

    def increment_request_count(self):
        self.total_requests += 1
        print(f"[DEBUG] Total requests: {self.total_requests}")

    def compute_error_rate(self):
        self.metrics["error_rate"] = (self.error_count / self.total_requests) * 100 if self.total_requests else 0
        print(f"[DEBUG] Error rate: {self.metrics['error_rate']}%")

    def record_rouge_scores(self, reference, hypothesis):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        self.metrics["rouge_scores"] = scores[0]
        print(f"[DEBUG] ROUGE scores: {scores[0]}")

    def generate_report(self):
        self.compute_error_rate()
        print(f"[DEBUG] Final metrics report generated: {self.metrics}")
        return self.metrics


# Example Summarization Workflow with Metrics Tracking
def test_summarization_workflow(video_id, reference_summary):
    tracker = MetricsTracker()
    recognizer = Recognizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("VortexKnight7/Video-Summ-Qwen")
    model = AutoModelForCausalLM.from_pretrained("VortexKnight7/Video-Summ-Qwen").to(device)
    print("[DEBUG] Model and tokenizer loaded successfully")

    try:
        tracker.increment_request_count()

        # Fetch Transcript
        print(f"[DEBUG] Fetching transcript for video ID: {video_id}")
        start_time = time.time()
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([x['text'] for x in transcript])
        end_time = time.time()
        tracker.record_execution_time("fetch_transcript", start_time, end_time)
        print(f"[DEBUG] Transcript fetched successfully. Length: {len(text)} characters")

        # Summarization
        print("[DEBUG] Starting summarization...")
        start_time = time.time()
        prompt = f"Query: Give me a brief summary on\n\nTranscript:\n{text}\n\nSummary:"
        inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=128, num_beams=1, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1].strip()
        end_time = time.time()
        tracker.record_model_latency(end_time - start_time)
        print(f"[DEBUG] Summary generated: {summary}")

        # Evaluate Summarization Quality
        print("[DEBUG] Evaluating summarization quality...")
        tracker.record_rouge_scores(reference_summary, summary)

        # Memory Usage
        tracker.record_memory_usage()

    except Exception as e:
        print(f"[ERROR] Error during processing: {e}")
        tracker.increment_error_count()

    return tracker.generate_report()


# Example Usage
if __name__ == "__main__":
    # Replace with your YouTube video ID and reference summary
    # video_id = "owlHiqL3nAI"  # Example: "dQw4w9WgXcQ"
    video_id = "pD2M5sLr3CQ"  # Example: "dQw4w9WgXcQ"

    reference_summary = "This is the expected summary of the video."

    print("[DEBUG] Starting performance analysis...")
    report = test_summarization_workflow(video_id, reference_summary)

    # Print the metrics report
    print("[DEBUG] Performance analysis completed. Final Report:")
    print(json.dumps(report, indent=4))
