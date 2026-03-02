# AI Courses

## TODOs
- Upgrade TTS to Qwen3 (maybe have alternative for test/prod?)

    - https://huggingface.co/spaces/Qwen/Qwen3-TTS
Prompt
Speak as a math professor, smart, wise, but compassionate, understanding that students dont currently posess all of his knowlege


- Fix a data pipeline from a concept, so that a model can use RAG, generate a script and pass it onto a model that generates the video


- Correlate color with beräkningar

## Pipeline Overview

```mermaid
flowchart TD
    A[1. Generate script for lesson] --> B[2. Generate Python file main which contains Manim and text]
    B --> C[3a. Generate video MP4<br>using Manim]
    B --> D[3b. Generate audio file<br>using audiomanager]
    C --> E[4. Merge audio and video]
    D --> E
```
