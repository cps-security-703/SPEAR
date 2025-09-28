# Llama 3.1-8B Agent

A conversational AI agent powered by Meta's Llama 3.1-8B model via Hugging Face.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify setup:**
   ```bash
   python test_agent.py
   ```

3. **Run the agent:**
   ```bash
   python llm_agent.py
   ```

## Features

- **Interactive Chat**: Continuous conversation loop
- **Hugging Face Integration**: Uses your token for model access
- **GPU Support**: Automatically uses CUDA if available
- **Memory Efficient**: Optimized for local deployment

## Usage

The agent will prompt you for input and respond using Llama 3.1-8B. Type `quit`, `exit`, or `bye` to stop.

## Requirements

- Python 3.8+
- ~16GB storage for model download (first run only)
- 8GB+ RAM recommended
- CUDA-compatible GPU recommended (optional)

## Token

Your Hugging Face token is stored in `keys.txt`. Keep this file secure and never commit it to version control.
