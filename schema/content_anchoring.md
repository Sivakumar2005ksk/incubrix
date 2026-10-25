# Content-Anchored Editing: Architectural Specification

## 1. The Core Problem with Clock-Based Video Editing
In conventional Non-Linear Editing systems (NLEs), video elements are placed at fixed clock timestamps (e.g., `01:40.00`). In conversational, prompt-based editing, this paradigm collapses immediately because instructions are applied iteratively:

- **Clock Drift:** If a user adds a caption on the phrase *"Welcome to the channel"* at `0:42` and subsequently says *"Cut the first 30 seconds"*, a clock-based system leaves the caption stranded at `0:42` (which now displays content from original minute `1:12`).
- **Cascading Distortions:** Combining speed changes (e.g. 2x speedup), silence removals, clip reorderings, and undo/redo operations compounds timestamp errors quadratically.

---

## 2. The Content-Anchored Solution
IncuBrix prompt-based video editing engine uses a **Dual-Coordinate System** that divorces content representation from clock playback.

```
       ┌─────────────────────────────────────────────────────────────┐
       │             IMMUTABLE SOURCE COORDINATES                    │
       │  Source File Hash (SHA-256)                                 │
       │  WordToken w_001 [0.00s - 0.45s]                            │
       │  WordToken w_042 [42.10s - 42.45s] ("Welcome")              │
       │  WordToken w_043 [42.45s - 42.80s] ("to")                   │
       │  WordToken w_044 [42.80s - 43.10s] ("the")                  │
       │  WordToken w_045 [43.10s - 43.50s] ("channel")              │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                         Content-Bound Edit Operation
                     AddCaption(text="Welcome!", anchor=[w_042..w_045])
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             NON-DESTRUCTIVE EVENT LOG                       │
       │  Op 1: AddCaption on [w_042..w_045]                         │
       │  Op 2: CutRange(source_start=0.0s, source_end=30.0s)        │
       │  Op 3: SpeedChange(source_start=60.0s, source_end=90.0s, 2x)│
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                     Pure Dynamic Projection Function f(Source, Ops)
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             PROJECTED PLAYHEAD TIMELINE                     │
       │  Clip 1: Source [30.0s - 60.0s]  --> Playhead [0.0s - 30.0s]│
       │  Caption: Attached to w_042..w_045 --> Playhead [12.1s - 13.5s]
       │  Clip 2: Source [60.0s - 90.0s] (2x)--> Playhead [30.0s - 45.0s]
       └─────────────────────────────────────────────────────────────┘
```

### Key Principles of Content Anchoring:
1. **Immutable Word Tokens:** During video ingestion, Speech-to-Text tokenizes audio into immutable `WordToken` records with unique identifiers (`w_001`, `w_002`, ...). Each token stores its immutable source start and end timestamps and diarized speaker.
2. **Anchor References:** Visual captions, bleeps, audio mutes, and speaker-specific operations attach explicitly to `WordToken.id` sets rather than playhead seconds.
3. **Pure Projection:** The output playhead timeline is a purely functional projection evaluated over the untouched source video and the active event log.
4. **Resilience to Cascading Edits:**
   - When the first 30 seconds are cut, tokens before `30.0s` are marked as excluded from the projected timeline.
   - The caption anchored to `[w_042, w_043, w_044, w_045]` recalculates its playhead position relative to the remaining active clips: `42.10s - 30.00s = 12.10s`. The caption automatically moves with the spoken words.
   - If the cut intro is subsequently restored via selective restore or undo, the projection re-includes the intro, and the caption naturally lands back at `42.10s`.

---

## 3. Boundary Snapping (Silence Protection)
Cuts that specify clock times or word boundaries automatically snap to the nearest silence interval:
- If a cut command targets `0.0s` to `30.0s`, and word token `w_035` spans `29.8s` to `30.2s`, blindly cutting at `30.0s` would slice the word in half.
- The state machine scans adjacent `SilenceSpan` markers and snaps the cut boundary to the silence gap following `w_035` (e.g., `30.25s`).
- This guarantees zero word clipping across all automated cuts.

---

## 4. Deterministic State Hashing
Every projection recalculates a canonical SHA-256 hash (`timeline_hash`) by serializing the ordered projected clips, speed curves, and visual overlays:
$$\text{timeline\_hash} = \text{SHA256}(\text{CanonicalJSON}(\text{ActiveProjection}))$$
Any identical sequence of operations on the same source file produces an identical `timeline_hash`, guaranteeing **100% determinism** across independent runs.
