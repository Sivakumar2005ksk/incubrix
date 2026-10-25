# AI Assistance Disclosure (AI_USE.md)

As required by Section 6 of the IncuBrix Engineering Assessment:

## 1. AI Coding Assistants Used
- **Antigravity (Advanced Agentic Assistant)**:
  - Assisted in scaffolding the microservice architecture and drafting Pydantic data schemas.
  - Assisted in formulating complex FFmpeg filtergraph strings (audio ducking sidechain compression and PTS alignment).
  - Assisted in writing test scenario fixtures for the 8-step worked example.

## 2. Models Used
- **Offline Deterministic Intent Matcher**: Default engine used for 100% deterministic, zero-cost, zero-API-key evaluation.
- **Google Gemini 1.5 Flash (Optional)**: Swappable adapter for conversational intent classification; translates natural language requests into structured parameters while all timeline projection remains deterministic code.
- **OpenAI GPT-4o-mini / Ollama LLaMA 3 (Optional)**: Alternative swappable adapters behind the common `BaseLLMAdapter` interface.
- All core logic (content anchoring math, interval trimming, undo/redo state stacks, and time policy translation) was formulated deterministically in code.
