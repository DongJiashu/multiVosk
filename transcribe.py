#!/usr/bin/env python3

# ==============================================================================
# AUDIO TRANSCRIPTION SYSTEM
# ==============================================================================
# Processes WAV files using multiple VOSK models, get pruned files, combines results, 
# and generates comprehensive reports with accuracy metrics. Automatically downloads 
# missing models before processing.
# Features:
# - Automatic model downloading
# - Multiple models transcription
# - get pruned files
# - Sentence-level voting combination
# - WER/CER calculation
# - Organized output structure

import os
import sys
import wave
import json
import re
import shutil
import subprocess
from vosk import Model, KaldiRecognizer, SetLogLevel
from difflib import SequenceMatcher
from datetime import datetime
from jiwer import wer, cer
from collections import defaultdict

SetLogLevel(-1)


# ==============================================================================
# MODEL MANAGEMENT
# ==============================================================================
# Handles checking for local models and launching downloader if needed

class VoskModelManager:
    MODEL_NAMES = [
        "vosk-small-en-us-0.15",
        "vosk-en-us-0.22", 
        "vosk-en-us-0.22-lgraph"
    ]
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def check_local_models(self):
        """Check which models exist locally"""
        local_models = {}
        for model_name in self.MODEL_NAMES:
            model_path = os.path.join(self.model_dir, model_name)
            if os.path.exists(model_path):
                local_models[model_name] = model_path
        return local_models
    
    def download_missing_models(self):
        """Launch downloader.py to get missing models"""
        try:
            subprocess.run(["python3", "downloader.py"], check=True)
            return self.check_local_models()
        except subprocess.CalledProcessError as e:
            print(f"Failed to download models: {e}")
            return {}

    def get_model_paths(self):
        """Get all model paths, downloading missing ones if needed"""
        local_models = self.check_local_models()
        
        if len(local_models) < len(self.MODEL_NAMES):
            print("Some models missing, downloading...")
            downloaded_models = self.download_missing_models()
            local_models.update(downloaded_models)
        
        return local_models


# ==============================================================================
# TRANSCRIBER CLASS
# ==============================================================================
# Handles audio transcription using multiple VOSK models with priority ordering

class EnhancedTranscriber:
    def __init__(self, model_paths):
        self.models = {}
        self.priority_order = [
            "vosk-en-us-0.22",
            "vosk-en-us-0.22-lgraph", 
            "vosk-small-en-us-0.15"
        ]
        
        for name in self.priority_order:
            path = model_paths.get(name)
            if path:
                try:
                    self.models[name] = Model(path)
                    print(f"Loaded model: {name}")
                except Exception as e:
                    print(f"Failed to load model {name}: {e}")

    @staticmethod
    def validate_audio(wav_path):
        try:
            with wave.open(wav_path, 'rb') as wf:
                return (wf.getnchannels() == 1 and 
                       wf.getsampwidth() == 2 and 
                       wf.getframerate() in [8000, 16000, 44100])
        except Exception as e:
            print(f"Error validating {wav_path}: {e}")
            return False

    def transcribe(self, audio_path):
        if not self.validate_audio(audio_path):
            return {}

        results = {}
        for model_name, model in self.models.items():
            wf = wave.open(audio_path, 'rb')
            rec = KaldiRecognizer(model, wf.getframerate())
            full_text = []
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    full_text.append(result.get('text', ''))
            
            final = json.loads(rec.FinalResult())
            full_text.append(final.get('text', ''))
            results[model_name] = ' '.join(filter(None, full_text))
            wf.close()
        
        return results

    def combine_results(self, model_results):
        if not model_results:
            return ""
            
        transcript_counts = defaultdict(int)
        for text in model_results.values():
            transcript_counts[text] += 1
        
        max_count = max(transcript_counts.values())
        candidates = [text for text, count in transcript_counts.items() if count == max_count]
        
        if len(candidates) == 1:
            return candidates[0]
            
        for model_name in self.priority_order:
            if model_name in model_results and model_results[model_name] in candidates:
                return model_results[model_name]
        
        return model_results[self.priority_order[0]]

    @staticmethod
    def clean_text(text):
        text = text.lower() #lowercase
        text = re.sub(r'[^\w\s]', '', text) #clean punctuation
        return re.sub(r'\s+', ' ', text).strip() #clean multiple whitespace
    
    #handle misspelling, threshold can be adjusted 
    def texts_match(self, text_a, text_b, threshold=0.8):
        return SequenceMatcher(None, 
                             self.clean_text(text_a), 
                             self.clean_text(text_b)).ratio() >= threshold


# ==============================================================================
# OUTPUT HANDLING
# ==============================================================================
# Manages file organization and archiving of transcription results

def _archive_good_match(original_file, file_id, reference_text, engine_name, model_output, pruned_dir):
    audio_output_dir = os.path.join(pruned_dir, engine_name, "wavs")
    text_output_dir = os.path.join(pruned_dir, engine_name, "texts")
    
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)

    output_audio_path = os.path.join(audio_output_dir, f"{file_id}.wav")
    if not os.path.exists(output_audio_path):
        shutil.copy2(original_file, output_audio_path)
    
    with open(os.path.join(text_output_dir, f"{file_id}.txt"), 'w', encoding='utf-8') as f:
        f.write(EnhancedTranscriber.clean_text(reference_text))


# ==============================================================================
# REPORT GENERATION
# ==============================================================================
# Creates concise transcription reports with accuracy metrics

def generate_report(results, output_dir, transcriber):
    report_path = os.path.join(output_dir, "report.txt")
    model_stats = defaultdict(lambda: {'wer': [], 'cer': []})
    combined_stats = {'wer': [], 'cer': []}

    with open(report_path, 'w') as f:
        for file_info in results:
            base_name = file_info['filename']
            f.write(f"File: {base_name}\n")
            # Insert reference line at the beginning of each file section
            reference = None
            for ref_model in transcriber.priority_order:
                ref_path = os.path.join(output_dir, "pruned", ref_model, "texts", f"{base_name}.txt")
                if os.path.exists(ref_path):
                    with open(ref_path, 'r') as rf:
                        reference = rf.read().strip()
                    break
            if reference:
                f.write(f"Reference: {reference}\n")

            for model, text in file_info['models'].items():
                line = f"{model}: {text}"
                raw_text_path = os.path.join(output_dir, "raw", model, f"{base_name}.txt")

                # Look for the reference from any pruned model
                reference = None
                for ref_model in transcriber.priority_order:
                    ref_path = os.path.join(output_dir, "pruned", ref_model, "texts", f"{base_name}.txt")
                    if os.path.exists(ref_path):
                        with open(ref_path, 'r') as rf:
                            reference = rf.read().strip()
                        break

                if reference and os.path.exists(raw_text_path):
                    with open(raw_text_path, 'r') as tf:
                        hypothesis = tf.read().strip()
                        ref_clean = transcriber.clean_text(reference)
                        hyp_clean = transcriber.clean_text(hypothesis)
                        current_wer = wer(ref_clean, hyp_clean)
                        current_cer = cer(ref_clean, hyp_clean)
                        line += f" [WER: {current_wer:.2f} | CER: {current_cer:.2f}"
                        if transcriber.texts_match(hypothesis, reference):
                            line += " | Match]"
                        else:
                            line += "]"
                        model_stats[model]['wer'].append(current_wer)
                        model_stats[model]['cer'].append(current_cer)
                f.write(line + "\n")

            if 'combined_result' in file_info:
                combined_text = file_info['combined_result']
                f.write(f"Combined: {combined_text}")
                reference = None
                for ref_model in transcriber.priority_order:
                    ref_path = os.path.join(output_dir, "pruned", ref_model, "texts", f"{base_name}.txt")
                    if os.path.exists(ref_path):
                        with open(ref_path, 'r') as rf:
                            reference = rf.read().strip()
                        break

                if reference:
                    ref_clean = transcriber.clean_text(reference)
                    combined_clean = transcriber.clean_text(combined_text)
                    current_wer = wer(ref_clean, combined_clean)
                    current_cer = cer(ref_clean, combined_clean)
                    f.write(f" [WER: {current_wer:.2f} | CER: {current_cer:.2f}")
                    if transcriber.texts_match(combined_text, reference):
                        f.write(" | Match]")
                    else:
                        f.write("]")
                    combined_stats['wer'].append(current_wer)
                    combined_stats['cer'].append(current_cer)
                f.write("\n")

            f.write("\n")

        f.write("=== Summary ===\n")
        for model, stats in model_stats.items():
            avg_wer = sum(stats['wer'])/len(stats['wer']) if stats['wer'] else 0
            avg_cer = sum(stats['cer'])/len(stats['cer']) if stats['cer'] else 0
            f.write(f"{model} - Avg WER: {avg_wer:.2f} | Avg CER: {avg_cer:.2f}\n")

        if combined_stats['wer']:
            avg_wer = sum(combined_stats['wer'])/len(combined_stats['wer'])
            avg_cer = sum(combined_stats['cer'])/len(combined_stats['cer'])
            f.write(f"Combined - Avg WER: {avg_wer:.2f} | Avg CER: {avg_cer:.2f}\n")


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
# Handles directory processing and workflow orchestration

def process_directory(input_dir, model_paths):
    # Load utterance file (for fuzzy matching against ASR outputs)
    utterance_list_path = os.path.join(input_dir, "utterance_file.txt")
    with open(utterance_list_path, 'r') as ufile:
        utterance_lines = [line.strip() for line in ufile if line.strip()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    raw_dir = os.path.join(output_dir, "raw")
    pruned_dir = os.path.join(output_dir, "pruned")
    result_dir = os.path.join(output_dir, "result")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(pruned_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    transcriber = EnhancedTranscriber(model_paths)
    results = []
    
    for wav_file in [f for f in os.listdir(input_dir) if f.endswith('.wav')]:
        base_name = os.path.splitext(wav_file)[0]
        audio_path = os.path.join(input_dir, wav_file)
        
        print(f"Processing {wav_file}...")
        
        model_results = transcriber.transcribe(audio_path)
        if not model_results:
            continue
        
        for model_name, text in model_results.items():
            model_raw_dir = os.path.join(raw_dir, model_name)
            os.makedirs(model_raw_dir, exist_ok=True)

            with open(os.path.join(model_raw_dir, f"{base_name}.txt"), "w") as f:
                f.write(text)

            # Match this output against utterance list
            best_match = None
            best_score = 0
            for utterance in utterance_lines:
                score = SequenceMatcher(None,
                                        transcriber.clean_text(text),
                                        transcriber.clean_text(utterance)).ratio()
                if score > best_score:
                    best_score = score
                    best_match = utterance

            if best_score >= 0.8:
                _archive_good_match(
                    audio_path,
                    base_name,
                    best_match,
                    model_name,
                    text,
                    pruned_dir
                )
        
        combined = transcriber.combine_results(model_results)
        if combined:
            with open(os.path.join(result_dir, f"{base_name}.txt"), "w") as f:
                f.write(combined)
        
        results.append({
            'filename': base_name,
            'models': model_results,
            'combined_result': combined
        })
    
    generate_report(results, output_dir, transcriber)
    print(f"Processing complete. Results saved to: {output_dir}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_enhanced.py <input_dir>")
        sys.exit(1)
    
    manager = VoskModelManager()
    model_paths = manager.get_model_paths()
    
    if not model_paths:
        print("Error: No VOSK models available")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)
    
    process_directory(input_dir, model_paths)