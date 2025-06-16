#!/usr/bin/env python3

import os
import requests
import zipfile

class VoskModelManager:
    """Handles VOSK model download and local storage"""
    
    MODEL_URLS = {
        "vosk-small-en-us-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "vosk-en-us-0.22": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "vosk-en-us-0.22-lgraph": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
    }
    
    def __init__(self, model_dir="models"):
        """Initialize with model storage directory"""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _download_model(self, model_name):
        """Download and extract model if not present locally"""
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.exists(model_path):
            return model_path
            
        print(f"Downloading model: {model_name}")
        url = self.MODEL_URLS[model_name]
        zip_path = os.path.join(self.model_dir, f"{model_name}.zip")
        
        try:
            # Download model zip
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract model
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.model_dir)
            
            os.remove(zip_path)  # Clean up zip file
            
            # Handle possible different extracted folder names
            if not os.path.exists(model_path):
                for f in os.listdir(self.model_dir):
                    if f.startswith("vosk-model-"):
                        os.rename(os.path.join(self.model_dir, f), model_path)
                        break
            
            return model_path
            
        except Exception as e:
            print(f"Failed to download/extract model {model_name}: {e}")
            return None

    def get_model_paths(self):
        """Get paths for all models, downloading if necessary"""
        model_paths = {}
        for model_name in self.MODEL_URLS.keys():
            path = self._download_model(model_name)
            if path:
                model_paths[model_name] = path
        return model_paths

if __name__ == "__main__":
    print("VOSK Model Downloader")
    print("This script downloads VOSK models for speech recognition.")
    print("Models will be downloaded to the 'models' directory.")
    
    manager = VoskModelManager()
    model_paths = manager.get_model_paths()
    
    if model_paths:
        print("\nSuccessfully downloaded models:")
        for name, path in model_paths.items():
            print(f"- {name}: {path}")
    else:
        print("No models were downloaded.")