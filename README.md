# Multi-Modal Toxicity Detector: AI Moderator 🛡️🤖
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Framework-green.svg)](https://flask.palletsprojects.com/)
[![Node.js](https://img.shields.io/badge/Node.js-Microservices-68a063.svg)](https://nodejs.org/)

[![MediaPipe](https://img.shields.io/badge/MediaPipe-ComputerVision-orange.svg)](https://developers.google.com/mediapipe)
[![Whisper](https://img.shields.io/badge/Whisper-AudioProcessing-blueviolet.svg)](https://github.com/openai/whisper)
[![Gemini](https://img.shields.io/badge/Gemini-AI_Core-red.svg)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ToxicBERT-yellow.svg)](https://huggingface.co/)

[![Socket.io](https://img.shields.io/badge/Socket.io-Realtime-010101.svg)](https://socket.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [System Interface & Dashboards](#-system-interface--dashboards)
3. [Core Architecture & Algorithm](#-core-architecture--algorithm)
4. [Key Features & Business Value](#-key-features--business-value)
5. [Technology Stack](#-technology-stack)
6. [Directory Structure](#-directory-structure)
7. [Local Installation](#-local-installation)
8. [API & Data Schemas](#-api--data-schemas)
9. [Future Roadmap](#-future-roadmap)

## 📖 Detailed Project Overview

The **Multimodal Toxicity Detector** is an enterprise-grade security intelligence system engineered to address the inherent limitations of conventional, text-only content moderation. In modern digital communication, toxic behavior has evolved beyond simple keyword usage; it now manifests through aggressive non-verbal cues, vocal intensity, and context-dependent sarcasm that legacy systems frequently misinterpret.

### 🧠 The Problem Statement
Existing moderation infrastructures operate in isolation, analyzing text streams independently of the underlying audio or visual behavior. This architectural silo leads to two systemic failure points:
1. **High False-Positive Rates**: Rigid keyword matching fails to distinguish between aggressive harassment and benign cultural slang, degrading user experience.
2. **Adversarial Evasion**: Malicious actors bypass legacy filters by using phonetic substitutions, manipulated audio, or non-verbal harassment that remains invisible to text-based detectors.

### 💡 Our Solution: The "Contextual Jury" Architecture
This project implements a **Triple-Stream Fusion Architecture** that synchronizes input from three distinct AI pipelines, aggregating them into a unified "Context Frame" before passing them to an LLM-based "Jury" for high-fidelity classification.

* **Behavioral Vision Stream (MediaPipe)**: Tracks facial landmarks and hand gestures in real-time. It maps kinematic data to aggressive posture modeling, flagging physical escalations before they manifest in text.
* **Acoustic Intelligence (OpenAI Whisper)**: Processes audio streams with high temporal precision, capturing not just the words spoken, but the underlying sentiment and vocal prosody.
* **Semantic Reasoning (Toxic-BERT)**: Serves as the primary heuristic filter, performing lightning-fast sentiment analysis to identify known malicious linguistic patterns.
* **The "Jury" Layer (Gemini-1.5-flash)**: The definitive decision node. By receiving aggregated visual metadata, full transcriptions, and BERT-derived sentiment scores, the LLM provides a nuanced, human-like contextual ruling. This eliminates the "black box" nature of traditional AI by providing an *explainable* reason for every flag.

### ⚙️ Engineering Philosophy & System Flow
Beyond the models, the system is designed for **High-Concurrency Low-Latency (HCLL) environments**:

* **Asynchronous Pipeline**: The architecture utilizes non-blocking I/O to ensure that vision, audio, and text streams are processed in parallel, preventing bottlenecking at the inference stage.
* **The Socket-Interface**: By leveraging **Flask-SocketIO**, the system maintains persistent, full-duplex communication with the client, ensuring that intervention (Warning/Mute/Block) happens in sub-100ms time.
* **Modularity via Policy Injection**: Instead of retraining heavy models, we utilize a dynamic "Policy Injection" layer. You can update the Jury's moderation guidelines (e.g., changing strictness for a gaming server vs. a professional meeting) simply by updating a JSON policy document, making the system highly adaptable to evolving digital environments.

### 🚀 Performance Highlights
* **Contextual Accuracy**: Achieves a significant reduction in false-positive rates by treating language as a multi-layered signal rather than a flat data point.
* **Scalability**: Designed for integration into large-scale streaming platforms, with a microservices-compatible architecture that allows the NLP and Computer Vision pipelines to scale independently based on load.
## 🖥 System Interface & Dashboards

The system features a **Unified Observability Dashboard** engineered for real-time decision-making and post-incident auditing. This interface acts as the central command center for the entire multimodal pipeline, bridging the gap between raw data inference and actionable moderation.

![System Dashboard](assets/dashboard.png)

### 📊 Core Dashboard Modules
* **Multimodal Stream Aggregator**: A side-by-side visualization feed rendering the active video stream (MediaPipe overlay), live audio waveforms, and a scrolling transcript. This enables operators to correlate visual posture with verbal content in real-time.
* **Intervention Status Engine**: An automated state-machine UI that updates enforcement status based on the "Jury" feedback loop:
    * 🟢 **Status: Secure** — Metrics within normal behavioral parameters.
    * 🟡 **Status: Warning** — Triggered by mild non-compliance; displays specific policy triggers.
    * 🔴 **Status: Enforced** — Active block/mute action with timestamp and violation details.
* **Explainability Panel (LLM-Logic)**: A specialized panel that renders the "Jury's Decision" from **Gemini-1.5-flash**. This provides a plain-text rationale for every flag, demystifying the AI decision-making process for platform moderators.

### ⚙️ Control & Configuration
* **Policy Threshold Overrides**: Allows administrators to dynamically toggle sensitivity levels for specific streams (e.g., adjusting thresholds for audio aggression vs. visual gestures).
* **Audit & Export Utility**: A built-in feature for incident logging. It generates structured summaries of flagged events—capturing visual frame contexts, audio transcripts, and the Jury’s explanation—formatted for regulatory compliance review.
* **System Health Telemetry**: Real-time monitoring of inference pipeline latency (ms) across Vision, Audio, and NLP streams to ensure performance remains within sub-100ms thresholds.
