# Sources & Third-Party Attributions

All libraries, models, and tools used in this project are strictly licensed for commercial use (MIT, Apache-2.0, BSD-3-Clause, LGPL-2.1+).

## 1. Libraries & Frameworks
| Library | Version | License | Purpose |
| :--- | :--- | :--- | :--- |
| **FastAPI** | >= 0.111.0 | MIT | Asynchronous web framework for HTTP microservice endpoints |
| **Uvicorn** | >= 0.30.0 | BSD-3-Clause | High-performance ASGI server |
| **Pydantic** | >= 2.7.0 | MIT | Data parsing, validation, and JSON schema generation |
| **PyYAML** | >= 6.0.1 | MIT | Parsing scripted session `.yaml` input files |
| **OpenTimelineIO** | >= 0.16.0 | Apache-2.0 | Standardized NLE interchange format export |
| **Boto3** | >= 1.34.0 | Apache-2.0 | S3/MinIO cloud storage adapter |
| **Pytest** | >= 8.2.0 | MIT | Unit, integration, and end-to-end test framework |
| **HTTPX** | >= 0.27.0 | BSD-3-Clause | Async HTTP client for webhook delivery and testing |

## 2. Video & Audio Engine
| Tool / Model | Version | License | Purpose |
| :--- | :--- | :--- | :--- |
| **FFmpeg** | 6.x / 7.x | LGPL-2.1+ / GPL | Audio-visual compositing, trims, speed scaling, ducking |
| **MockSTT** | 1.0.0 (Internal) | MIT | Deterministic zero-key speech-to-text token provider |
| **Faster-Whisper** | (Optional adapter) | MIT | Local open-source CTranslate2 Whisper speech-to-text |
| **Google Gemini API** | 1.5 Flash / Pro (Optional) | Commercial API Terms | Optional natural language intent classification adapter |
| **OpenAI API** | GPT-4o-mini (Optional) | Commercial API Terms | Optional natural language intent classification adapter |
| **Ollama** | Local LLaMA 3 (Optional) | LLaMA 3 Community License | Optional on-device local intent classification adapter |

## 3. Data & Media Assets
- Test fixtures and synthetic word transcripts are synthesized purely in code under the **MIT License**.
- No unauthorized likenesses, copyrighted music, or unconsented voices are included.
