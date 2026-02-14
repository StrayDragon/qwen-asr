#!/usr/bin/env python3
"""
Simple test client for the OpenAI-compatible transcription API.
"""
import os
import sys
from pathlib import Path

import httpx


# Default configuration
API_URL = os.getenv("QWEN_API_URL", "http://localhost:8011")
API_TOKEN = os.getenv("QWEN_API_TOKEN", "sk-test-key")
MODEL = os.getenv("QWEN_MODEL", "qwen-asr-0.6b")


def test_non_streaming(audio_path: str):
    """Test non-streaming transcription."""
    print(f"Testing non-streaming transcription with {audio_path}...")
    
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "audio/wav")}
        data = {
            "model": MODEL,
            "stream": "false",
        }
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
        }
        
        response = httpx.post(
            f"{API_URL}/v1/audio/transcriptions",
            files=files,
            data=data,
            headers=headers,
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResult:")
        print(f"  Text: {result['text']}")
        print(f"  Usage: {result['usage']}")
        return result
    else:
        print(f"\nError: {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)


def test_streaming(audio_path: str):
    """Test streaming transcription with SSE."""
    print(f"Testing streaming transcription with {audio_path}...")
    
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "audio/wav")}
        data = {
            "model": MODEL,
            "stream": "true",
        }
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
        }
        
        with httpx.stream(
            "POST",
            f"{API_URL}/v1/audio/transcriptions",
            files=files,
            data=data,
            headers=headers,
        ) as response:
            
            if response.status_code != 200:
                print(f"\nError: {response.status_code}")
                print(f"  {response.text}")
                sys.exit(1)
            
            print("\nStreaming events:")
            for line in response.iter_lines():
                if line.startswith("data: "):
                    import json
                    event = json.loads(line[6:])  # Strip "data: " prefix
                    event_type = event.get("type", "")
                    if event_type == "transcript.text.delta":
                        print(event["delta"], end="", flush=True)
                    elif event_type == "transcript.text.done":
                        print(f"\n\nDone. Usage: {event['usage']}")
                        return event
    
    return None


def test_list_models():
    """Test list models endpoint."""
    print("Testing /v1/models endpoint...")
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
    }
    
    response = httpx.get(
        f"{API_URL}/v1/models",
        headers=headers,
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nAvailable models:")
        for model in result.get("data", []):
            print(f"  - {model['id']}")
        return result
    else:
        print(f"\nError: {response.status_code}")
        print(f"  {response.text}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test client for Qwen ASR API")
    parser.add_argument("audio", nargs="?", help="Audio file path")
    parser.add_argument("--stream", action="store_true", help="Test streaming mode")
    parser.add_argument("--models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.models:
        test_list_models()
    elif args.audio:
        if args.stream:
            test_streaming(args.audio)
        else:
            test_non_streaming(args.audio)
    else:
        parser.print_help()
        sys.exit(1)
