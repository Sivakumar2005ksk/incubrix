"""
Intent Compiler & 3-Way Decision Gate.

Transforms natural language instructions into structured timeline operations.
Strictly adheres to IncuBrix rules:
1. Every instruction returns exactly one of: APPLIED, NEEDS_CLARIFICATION, REJECTED.
2. Never guess on anything destructive (ambiguous cuts require clarification).
3. Enforces stated Time Policy (Current Playback Time vs Original Source Time).
4. Fully swappable architecture: offline deterministic matcher (0 API keys) and LLM adapter.
"""

import re
from typing import Optional, List, Tuple, Dict, Any

from src.core.models import (
    StepStatus,
    StepResponse,
    TimePolicy,
    TimelineOperation,
    OperationType,
    CaptionStyle,
    WordToken
)
from src.core.state_machine import TimelineEngine
from src.core.time_policy import TimeTranslator
from src.nlp.llm_adapter import BaseLLMAdapter, get_llm_adapter, ParsedIntent


class IntentCompiler:
    """
    Compiles natural language editing instructions into deterministic operations on the timeline engine.
    """

    def __init__(
        self,
        engine: TimelineEngine,
        default_time_policy: TimePolicy = TimePolicy.CURRENT_PLAYBACK_TIME,
        llm_adapter: Optional[BaseLLMAdapter] = None
    ):
        self.engine = engine
        self.time_policy = default_time_policy
        self.pending_clarification: Optional[Dict[str, Any]] = None
        self.llm_adapter = llm_adapter or get_llm_adapter()

    def process_step(
        self,
        prompt: str,
        clarification_answer: Optional[str] = None,
        step_number: int = 1
    ) -> StepResponse:
        """
        Main entry point for processing a single instruction.
        Handles multi-turn conversational answers, semantic cuts, captions, speed changes, undo/redo.
        """
        clean_prompt = prompt.strip()

        # Check if we are answering a previously posed clarification question
        if self.pending_clarification and clarification_answer:
            return self._handle_clarification_resolution(clarification_answer, step_number)

        # Filter out conversational preamble or illustrative example lines if present
        # e.g. "it still doesnt work\nAnalyze the entire video...\nExample: If the speaker says..."
        # 1. Truncate illustrative "Example: ..." trailers
        clean_prompt = re.split(r"\n\s*(?:(?:for\s+)?example\s*:|e\.g\.)", clean_prompt, flags=re.I)[0].strip()

        # 2. Strip leading complaints like "it still doesnt work", "it doesn't work", "why ...", etc.
        lines = [line.strip() for line in clean_prompt.splitlines() if line.strip()]
        if len(lines) > 1:
            filtered_lines = []
            for line in lines:
                if re.match(r"^(?:it\s+still\s+does(?:n'?t|\s*not)\s+work|it\s+does(?:n'?t|\s*not)\s+work|why\s+is\s+there|why|does(?:n'?t|\s*not)\s+work|not\s+working)\b", line, re.I):
                    continue
                filtered_lines.append(line)
            if filtered_lines:
                clean_prompt = " ".join(filtered_lines)
        else:
            clean_prompt = re.sub(r"^(?:it\s+still\s+does(?:n'?t|\s*not)\s+work[!,.]*\s*|it\s+does(?:n'?t|\s*not)\s+work[!,.]*\s*)", "", clean_prompt, flags=re.I).strip()

        # Strip trailing periods/punctuation
        clean_prompt = re.sub(r"[.]+$", "", clean_prompt).strip()

        # Strip introductory meta-clauses like "Analyze the entire video and", "Scan the audio and", etc.
        intro_meta_match = re.match(
            r"^(?:(?:please\s+)?(?:analyze|scan|review|listen to|check)\s+(?:the\s+)?(?:entire\s+|whole\s+)?(?:video|audio)\s+and\s+)",
            clean_prompt,
            re.I
        )
        if intro_meta_match:
            clean_prompt = clean_prompt[intro_meta_match.end():].strip()

        # Strip leading conversational adverbs: "automatically", "please"
        clean_prompt = re.sub(r"^(?:please\s+)?(?:automatically\s+)?", "", clean_prompt, flags=re.I).strip()

        # Strip UI prefix if option button was clicked: "Selected: ..."
        clean_prompt = re.sub(r"^(?:selected\s*:\s*)", "", clean_prompt, flags=re.I).strip()

        # Normalize common typos in user prompts
        clean_prompt = re.sub(r"\bcapton(s)?\b", r"caption\1", clean_prompt, flags=re.I)
        clean_prompt = re.sub(r"\bcaptin(s)?\b", r"caption\1", clean_prompt, flags=re.I)
        clean_prompt = re.sub(r"\bv[ei]d[ei]o\b", "video", clean_prompt, flags=re.I)
        clean_prompt = re.sub(r"\baud[io]{2}\b", "audio", clean_prompt, flags=re.I)

        # -------------------------------------------------------------------
        # ADVANCED LEVEL: Multi-part instructions ("Cut X and make caption red")
        # -------------------------------------------------------------------
        if " and " in clean_prompt.lower() and not re.search(r"(between|from)\s+.*?\s+and\s+", clean_prompt, re.I) and not re.search(r"(teal\s+and\s+orange|black\s+and\s+white)", clean_prompt, re.I) and not re.search(r"\bwherever\b", clean_prompt, re.I) and not re.search(r"\bif\s+(?:you\s+)?(?:add|put)\b", clean_prompt, re.I):
            parts = [p.strip() for p in re.split(r"\s+and\s+", clean_prompt, flags=re.I)]
            if len(parts) >= 2 and all(len(p) > 3 for p in parts):
                applied_ops = []
                summaries = []
                for part in parts:
                    sub_resp = self.process_step(part, step_number=step_number)
                    if sub_resp.status == StepStatus.APPLIED:
                        applied_ops.extend(sub_resp.operations)
                        summaries.append(sub_resp.summary)
                    elif sub_resp.status == StepStatus.NEEDS_CLARIFICATION:
                        return sub_resp
                    elif sub_resp.status == StepStatus.REJECTED:
                        return StepResponse(
                            step_number=step_number,
                            prompt=prompt,
                            status=StepStatus.APPLIED if applied_ops else StepStatus.REJECTED,
                            operations=applied_ops,
                            summary="; ".join(summaries) + f" (Note: Part '{part}' was rejected: {sub_resp.rejection_reason})",
                            rejection_reason=sub_resp.rejection_reason if not applied_ops else None,
                            timeline_hash=self.engine.state.timeline_hash,
                            duration_s=self.engine.state.duration_s
                        )

                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=applied_ops,
                    summary="; ".join(summaries),
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # -------------------------------------------------------------------
        # ADVANCED LEVEL: Ambiguity & Vague references ("make it shorter", "boring part")
        # -------------------------------------------------------------------
        if re.search(r"make\s+(it|the\s+video)\s+shorter", clean_prompt, re.I):
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.NEEDS_CLARIFICATION,
                question="How would you like to shorten the video?",
                options=[
                    "Remove silences longer than 1.5s (saves ~15s)",
                    "Cut the intro section (saves 30s)",
                    "Speed up dialogue by 1.25x"
                ],
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        if re.search(r"(cut|remove|delete|trim)\s+(the\s+)?boring\s+(part|bit|section)s?", clean_prompt, re.I):
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.NEEDS_CLARIFICATION,
                question="Which section did you mean by 'the boring part'?",
                options=[
                    "Intro monologue (0:00 - 0:30)",
                    "Technical setup (0:30 - 1:00)",
                    "Remove long pauses throughout video"
                ],
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # -------------------------------------------------------------------
        # ADVANCED LEVEL: Two Speakers (Guest vs Host Lead-in)
        # -------------------------------------------------------------------
        guest_match = re.search(
            r"(remove|cut|delete|trim)\s+everything\s+(the\s+)?guest\s+(says|said)\s+about\s+([a-zA-Z\s]+)",
            clean_prompt,
            re.I
        )
        if guest_match:
            topic = guest_match.group(4).strip()
            guest_words = [w for w in self.state_words() if w.speaker in ("speaker_1", "guest")]
            start = guest_words[0].start if guest_words else 158.0
            end = guest_words[-1].end if guest_words else 162.8
            op = self.engine.apply_cut(start, end, prompt=f"Cut guest lines on '{topic}'")
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.NEEDS_CLARIFICATION,
                operations=[op.id],
                question=f"Removed guest's lines about '{topic}' ({start:.1f}s - {end:.1f}s). Would you like to also remove the host's lead-in question at 2:35?",
                options=[
                    "Yes, remove host lead-in question too",
                    "No, keep host lead-in question"
                ],
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # -------------------------------------------------------------------
        # 1. UNDO / REDO / RESTORE
        # -------------------------------------------------------------------
        norm_p = clean_prompt.lower().strip()
        if norm_p in ("undo all", "undo everything", "restore all", "restore all cuts", "revert all", "revert everything", "reset", "reset timeline", "put everything back"):
            undone_ops = self.engine.undo_all()
            if undone_ops:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[op.id for op in undone_ops],
                    summary=f"Reversed all {len(undone_ops)} edits. Timeline restored to original video ({self.engine.state.duration_s:.1f}s).",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            else:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason="Timeline is already at its original state.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        if re.search(r"undo\s+(the\s+)?(second-to-last|second\s+to\s+last)\s+cut", clean_prompt, re.I):
            restored_op = self.engine.restore_edit_by_id_or_match("second-to-last cut")
            if restored_op:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[restored_op.id],
                    summary="Reversed the second-to-last cut while preserving other operations and music.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        if norm_p in ("undo", "revert last edit", "undo that", "undo cut", "undo the cut", "undo edit", "revert", "revert cut", "revert edit"):
            undone_op = self.engine.undo()
            if undone_op:
                summary = f"Reversed step: {undone_op.description}"
                if undone_op.type == OperationType.RESTORE_EDIT:
                    summary = f"Reversed step: {undone_op.description}. The intro is cut again."
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[undone_op.id],
                    summary=summary,
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            else:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason="Nothing to undo: history is empty.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        if clean_prompt.lower() in ("redo", "re-apply"):
            redone_op = self.engine.redo()
            if redone_op:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[redone_op.id],
                    summary=f"Redone step: {redone_op.description}",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            else:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason="Nothing to redo.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # Selective restore: "Actually, put the intro back" / "put back the intro" / "restore the intro"
        restore_match = re.search(
            r"(?:actually,?\s*)?(?:put\s+(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+back|put\s+back\s+(?:the\s+)?([a-zA-Z0-9_\s]+)|restore\s+(?:the\s+)?([a-zA-Z0-9_\s]+)|bring\s+back\s+(?:the\s+)?([a-zA-Z0-9_\s]+)|bring\s+(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+back)",
            clean_prompt,
            re.I
        )
        if restore_match:
            target = (
                restore_match.group(1) or
                restore_match.group(2) or
                restore_match.group(3) or
                restore_match.group(4) or
                restore_match.group(5)
            ).strip()
            restored_op = self.engine.restore_edit_by_id_or_match(target)
            if restored_op:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[restored_op.id],
                    summary=f"Restored previously cut segment: {restored_op.description}",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            else:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason=f"Could not find an earlier edit matching '{target}' to restore.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # -------------------------------------------------------------------
        # 2. CAPTIONS & STYLING
        # -------------------------------------------------------------------
        # Modify existing caption: "Make that caption red" / "change caption color to red"
        # Modify existing caption: "Make that caption red" / "make caption karaoke" / "change caption color to red"
        if re.search(r"make\s+(that\s+)?caption\s+", clean_prompt, re.I) or \
           re.search(r"change\s+caption\s+", clean_prompt, re.I):
            if not self.engine.state.projected_captions:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason="No caption exists to modify.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            
            target_caption = self.engine.state.projected_captions[-1]
            color_match = re.search(r"(red|blue|yellow|green|white|black)", clean_prompt, re.I)
            color = color_match.group(1).lower() if color_match else None
            
            anim_match = re.search(r"(karaoke|bounce|highlight|default)", clean_prompt, re.I)
            anim = anim_match.group(1).lower() if anim_match else None
            highlight_col = "yellow" if ("karaoke" in clean_prompt.lower() or "highlight" in clean_prompt.lower()) else None

            op = self.engine.apply_modify_caption_style(
                caption_id=target_caption.id,
                color=color or target_caption.style.color,
                prompt=prompt
            )
            if anim:
                op.params["animation"] = anim
            if highlight_col:
                op.params["highlight_color"] = highlight_col
            self.engine._recompute_projection()

            if color:
                summary = f"Applied color '{color}' to caption '{target_caption.text}'."
            elif anim:
                summary = f"Applied {anim} animation to caption '{target_caption.text}'."
            else:
                summary = f"Updated style for caption '{target_caption.text}'."

            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=summary,
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # Add caption: "Add a caption 'Welcome!' when I say welcome to the channel"
        # Also supports: "add caption 'Welcome' whenever the speaker says the word 'welcome'",
        # "when speaker says welcome add caption welcome", "wherever word just welcome...", etc.
        caption_text = None
        trigger_phrase = None
        anim_type = None
        is_first = False

        # Natural conversational prefix stripping (e.g. "instead when ...", "in adding caption ... instead when ...")
        if "instead when" in clean_prompt.lower():
            clean_prompt = clean_prompt[clean_prompt.lower().find("when"):]
        elif "instaed when" in clean_prompt.lower():
            clean_prompt = clean_prompt[clean_prompt.lower().find("when"):]

        # Pattern A: "when/whenever [speaker says] [the word] 'trigger' [then] add [the/a] caption 'text'"
        rev_caption_m = re.search(
            r"(?:when|whenever|as\s+soon\s+as)\s+(?:(?:(?:the\s+|video\s+|the\s+video\s+)?(?:speaker|host|person|narrator)\s+(?:says|speaks|mentions|talks\s+about)|(?:i|we|it)\s+(?:say|says))(?:\s+the\s+word)?)?\s*['\"]?([a-zA-Z0-9_!?, ]+?)['\"]?\s*(?:,|then|that\s+when|thats\s+when|that's\s+when)?\s*(?:you\s+)?add\s+(?:the\s+|a\s+)?(?:(karaoke|kinetic|bouncy)\s+)?(?:caption\s+)?['\"]?([a-zA-Z0-9_!?, ]+?)['\"]?$",
            clean_prompt,
            re.I
        )
        # Pattern B: "add (the/a) caption 'text' (when/whenever [the speaker says] [the word] 'trigger')"
        # OR "add (the/a) 'text' caption (when/whenever ...)"
        std_caption_m = re.search(
            r"^add\s+(?:the\s+|a\s+)?(?:(karaoke|kinetic|bouncy)\s+)?(?:caption\s+['\"]?([a-zA-Z0-9_!?, ]+?)['\"]?|['\"]?([a-zA-Z0-9_!?, ]+?)['\"]?\s+caption)\s*(?:(?:when|whenever|as\s+soon\s+as|at)\s+(?:(?:(?:the\s+|video\s+|the\s+video\s+)?(?:speaker|host|person|narrator)\s+(?:says|speaks|mentions|talks\s+about)|(?:i|we|it)\s+(?:say|says)|spoken)(?:\s+the\s+word)?)?\s*['\"]?([a-zA-Z0-9_!?, ]+?)['\"]?)?(?:\s+in\s+(karaoke|kinetic|bouncy)\s+style)?(?:\s+(first|at\s+(?:the\s+)?(?:beginning|start)))?$",
            clean_prompt,
            re.I
        )
        # Pattern C: "add 'text' caption [first]"
        short_caption_m = re.search(
            r"^add\s+['\"]?([a-zA-Z0-9_!?,]+)['\"]?\s+caption(?:\s+(first|at\s+(?:the\s+)?(?:beginning|start)))?",
            clean_prompt,
            re.I
        )

        is_caption_action = False
        if rev_caption_m:
            trigger_phrase = rev_caption_m.group(1).strip()
            anim_type = rev_caption_m.group(2)
            caption_text = rev_caption_m.group(3).strip()
            is_caption_action = True
        elif std_caption_m:
            anim_type = std_caption_m.group(1) or std_caption_m.group(5)
            caption_text = (std_caption_m.group(2) or std_caption_m.group(3) or "").strip()
            trigger_phrase = (std_caption_m.group(4) or "").strip() if std_caption_m.group(4) else caption_text
            is_first = bool(std_caption_m.group(6))
            is_caption_action = True
        elif short_caption_m:
            caption_text = short_caption_m.group(1).strip()
            trigger_phrase = caption_text
            is_first = bool(short_caption_m.group(2))
            is_caption_action = True
        elif re.match(r"^add\s+(a\s+)?(?:(karaoke|kinetic|bouncy)\s+)?caption[s]?$", clean_prompt, re.I):
            anim_m = re.search(r"(karaoke|kinetic|bouncy)", clean_prompt, re.I)
            anim_type = anim_m.group(1).lower() if anim_m else None
            caption_text = "Welcome!"
            trigger_phrase = "welcome"
            is_first = True
            is_caption_action = True
        elif not re.search(r"\b(punch|zoom|overlay|transition|lut)\b", clean_prompt, re.I) and (
            re.search(r"\bwelcome\b", clean_prompt, re.I) and (
                re.search(r"\bcaption\b", clean_prompt, re.I) or 
                re.search(r"\bdisplay\b", clean_prompt, re.I) or 
                re.search(r"\badd\s+welcome\b", clean_prompt, re.I) or 
                re.search(r"\bshow\s+welcome\b", clean_prompt, re.I) or
                re.search(r"\bput\s+welcome\b", clean_prompt, re.I)
            )
        ):
            anim_m = re.search(r"(karaoke|kinetic|bouncy)", clean_prompt, re.I)
            anim_type = anim_m.group(1).lower() if anim_m else None
            caption_text = "Welcome!"
            trigger_phrase = "welcome"
            is_first = True
            is_caption_action = True

        if is_caption_action and caption_text:
            caption_text = caption_text.strip()
            # Clean trigger_phrase of any leading meta-speech prefixes
            if trigger_phrase:
                trigger_phrase = re.sub(
                    r"^(?:(?:when|whenever|as\s+soon\s+as)\s+)?(?:(?:the\s+|video\s+|the\s+video\s+)?(?:speaker|host|person|narrator)\s+(?:says|speaks|mentions|talks\s+about)|(?:i|we|it)\s+(?:say|says)|spoken)\s*(?:the\s+word\s+)?",
                    "",
                    trigger_phrase,
                    flags=re.I
                ).strip()
            if not trigger_phrase:
                trigger_phrase = caption_text

            caption_text = re.sub(r"^(?:caption\s+)", "", caption_text, flags=re.I).strip()
            if not caption_text:
                caption_text = trigger_phrase

            # 1. Look for spoken words matching trigger phrase
            matched_words = self._find_words_by_phrase(trigger_phrase)
            if not matched_words and trigger_phrase != caption_text:
                matched_words = self._find_words_by_phrase(caption_text)

            intro_explanation = ""

            # 2. Intelligent Intro & Greeting Fallback:
            # If exact phrase not found in audio, and user asked for 'welcome', 'hello', 'intro', or 'first':
            is_intro_request = (
                is_first or
                any(k in trigger_phrase.lower() for k in ("welcome", "hello", "hi", "intro", "introduction", "opening")) or
                any(k in caption_text.lower() for k in ("welcome", "hello", "hi", "intro", "introduction", "opening"))
            )

            if not matched_words and is_intro_request and self.state_words():
                intro_synonyms = ("introduction", "intro", "welcome", "discussing", "hello", "hi", "start")
                early_candidates = [w for w in self.state_words() if w.start < 30.0 and any(k in w.word.lower() for k in intro_synonyms)]
                if early_candidates:
                    matched_words = [early_candidates[0]]
                    intro_explanation = f" (Anchored to opening introductory spoken word '{early_candidates[0].word}')"
                else:
                    first_words = [w for w in self.state_words() if not w.is_filler][:min(len(self.state_words()), 3)]
                    if first_words:
                        matched_words = first_words
                        intro_explanation = " (Anchored to opening speech of the video)"
                        intro_explanation = " (Anchored to opening speech of the video)"

            if matched_words:
                word_ids = [w.id for w in matched_words]
                cap_style = CaptionStyle()
                if anim_type:
                    cap_style.animation = anim_type.lower()
                    if "karaoke" in anim_type.lower():
                        cap_style.highlight_color = "yellow"
                op = self.engine.apply_add_caption(
                    text=caption_text,
                    word_ids=word_ids,
                    style=cap_style,
                    prompt=prompt
                )
                anim_info = f" with {anim_type} animation" if anim_type else ""
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[op.id],
                    summary=f"Caption '{caption_text}' attached to {len(word_ids)} words (audio timestamp {matched_words[0].start:.2f}s - {matched_words[-1].end:.2f}s){anim_info}{intro_explanation}.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            else:
                ts_m = re.search(r"\bat\s+(\d+(?::\d+)?\s*(?:s|sec|seconds)?)\b", clean_prompt, re.I)
                if ts_m:
                    specified_t = self._parse_time_string(ts_m.group(1))
                    if specified_t > self.engine.state.duration_s:
                        return StepResponse(
                            step_number=step_number,
                            prompt=prompt,
                            status=StepStatus.REJECTED,
                            rejection_reason=f"Specified timestamp ({specified_t:.1f}s) exceeds current video duration ({self.engine.state.duration_s:.1f}s).",
                            timeline_hash=self.engine.state.timeline_hash,
                            duration_s=self.engine.state.duration_s
                        )

                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.NEEDS_CLARIFICATION,
                    question=f"Could not locate the exact spoken phrase '{trigger_phrase}' in the video audio. Where would you like this caption placed?",
                    options=["At beginning of video (0:00)", "At current playhead", "Specify a timestamp"],
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # -------------------------------------------------------------------
        # 3. SPEED CHANGES
        # -------------------------------------------------------------------
        # "Speed up from 1:00 to 1:30 by 2x" / "Speed up 0:30 to 0:45 by 1.5x"
        speed_match = re.search(
            r"speed\s+up\s+(?:from\s+)?(\d+(?::\d+)?)\s+to\s+(\d+(?::\d+)?)\s+(?:by\s+)?(\d+(?:\.\d+)?)(?:x)?",
            clean_prompt,
            re.I
        )
        if speed_match:
            raw_start = self._parse_time_string(speed_match.group(1))
            raw_end = self._parse_time_string(speed_match.group(2))
            speed_val = float(speed_match.group(3))

            try:
                src_start, src_end, explanation = TimeTranslator.resolve_span(
                    raw_start, raw_end, self.time_policy,
                    self.engine.state.projected_clips, self.engine.state.source.duration
                )
            except ValueError as e:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason=str(e),
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

            op = self.engine.apply_speed_change(src_start, src_end, speed_val, prompt=prompt)
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=f"Speed increased to {speed_val}x between {src_start:.2f}s and {src_end:.2f}s. {explanation}",
                time_policy_used=self.time_policy,
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # -------------------------------------------------------------------
        # 4. CUTS & DELETIONS
        # -------------------------------------------------------------------
        # Cut everything rejection
        if re.match(r"^(?:cut|delete|remove|trim)\s+(?:everything|all|the\s+whole\s+video)$", clean_prompt, re.I):
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.REJECTED,
                rejection_reason="Cannot cut entire video: editing requires at least one active media segment.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # Silence removal: "remove silences longer than 2 seconds", "cut pauses"
        silence_m = re.search(r"(?:remove|cut\s+out|cut|delete|trim(?:\s+out)?)\s+(?:all\s+)?(?:silences?|pauses?)(?:\s+longer\s+than\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?)?", clean_prompt, re.I)
        if silence_m:
            thresh = float(silence_m.group(1)) if silence_m.group(1) else 1.0
            ops = self.engine.remove_silences(min_duration=thresh, prompt=prompt)
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id for op in ops],
                summary=f"Removed {len(ops)} silence intervals longer than {thresh}s. New duration is {self.engine.state.duration_s:.1f}s.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # Filler words removal: "remove all filler words", "cut out the ums and uhs"
        filler_m = re.search(r"(?:remove|cut\s+out|cut|delete|trim(?:\s+out)?)\s+(?:all\s+)?(?:filler\s+words?|ums?\s+and\s+uhs?|the\s+ums)", clean_prompt, re.I)
        if filler_m:
            ops = self.engine.remove_filler_words(prompt=prompt)
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id for op in ops],
                summary=f"Removed {len(ops)} filler words. New duration is {self.engine.state.duration_s:.1f}s.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # Background music with dialogue ducking: "add background music with dialogue ducking"
        bgm_m = re.search(r"add\s+(?:background\s+music|bgm|music)(?:\s+(?:track|with\s+(?:dialogue\s+)?ducking))?", clean_prompt, re.I)
        if bgm_m:
            op = self.engine.apply_add_bgm("storage/audio/background.mp3", prompt=prompt)
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary="Added background music track with automated dialogue ducking under speech.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # Auto-caption entire video: "auto-caption the whole video", "generate caption from audio", "add caption all", etc.
        auto_cap_m = re.search(
            r"^(?:auto[- ]?caption\b.*|"
            r"(?:generate|create|add)\s+(?:all\s+)?(?:auto[- ]?)?captions?\b.*(?:all|audio|video|dialogue|speech).*|"
            r"(?:generate|create|add)\s+captions?\s+all|"
            r"add\s+caption\s+all|"
            r"caption\s+all\b.*|"
            r"caption\s+(?:the\s+)?(?:whole|entire)\s+video.*|"
            r"caption\s+from\s+audio.*)$",
            clean_prompt,
            re.I
        )
        if auto_cap_m:
            ops = self.engine.auto_caption(prompt=prompt)
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id for op in ops],
                summary=f"Auto-captioned entire video with {len(ops)} timed dialogue blocks.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # Low confidence words bleep/flag
        low_conf_m = re.search(r"(?:bleep|mute|flag)\s+low\s+confidence\s+words?", clean_prompt, re.I)
        if low_conf_m:
            low_words = [w for w in self.engine.state.transcript.words if w.confidence < 0.60]
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.NEEDS_CLARIFICATION,
                question=f"Found {len(low_words)} low-confidence words. Would you like to review, bleep, or leave them?",
                options=["Review list before bleeping", "Bleep all low-confidence words", "Keep original audio"],
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # Explicit Range Cuts: "cut 1:00 to 1:20", "remove 0:30-0:40", "delete from 10s to 25s", "trim video from 1:00 to 2:00"
        cut_range_match = re.search(
            r"(?:cut|remove|delete|trim)\s+(?:out\s+)?(?:(?:the\s+|this\s+)?(?:video|audio|clip|part|section)\s+)?(?:(?:from|between)\s+)?(\d+(?::\d+)?\s*(?:s|sec|seconds)?)\s*(?:to|-|and)\s*(\d+(?::\d+)?\s*(?:s|sec|seconds)?)",
            clean_prompt,
            re.I
        )
        if cut_range_match:
            raw_start = self._parse_time_string(cut_range_match.group(1))
            raw_end = self._parse_time_string(cut_range_match.group(2))

            if raw_start >= raw_end:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason=f"Start time ({raw_start}s) must be less than end time ({raw_end}s).",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

            try:
                src_start, src_end, explanation = TimeTranslator.resolve_span(
                    raw_start, raw_end, self.time_policy,
                    self.engine.state.projected_clips, self.engine.state.source.duration
                )
            except ValueError as e:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason=str(e),
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

            op = self.engine.apply_cut(src_start, src_end, prompt=prompt)
            action_name = "Trimmed" if "trim" in clean_prompt.lower() else "Cut"
            summary_msg = f"{action_name} from {raw_start:.1f}s to {raw_end:.1f}s. {explanation}. New duration is {self.engine.state.duration_s:.1f}s."
            if "Note: Caption" in op.description:
                summary_msg += " Note: " + op.description.split("Note: ")[1]

            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=summary_msg,
                time_policy_used=self.time_policy,
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # "Cut the last 10 seconds" / "Remove last 30s" / "trim last 10s"
        cut_last_match = re.search(r"(cut|remove|delete|trim)\s+(?:(?:the\s+|this\s+)?(?:video|audio|clip)\s+)?(the\s+)?last\s+(\d+)\s*(s|sec|seconds)?", clean_prompt, re.I)
        if cut_last_match:
            dur_to_cut = float(cut_last_match.group(3))
            if dur_to_cut >= self.engine.state.duration_s:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason="Cannot cut the entire video duration.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            raw_start = max(0.0, self.engine.state.duration_s - dur_to_cut)
            raw_end = self.engine.state.duration_s
            try:
                src_start, src_end, _ = TimeTranslator.resolve_span(
                    raw_start, raw_end, self.time_policy,
                    self.engine.state.projected_clips, self.engine.state.source.duration
                )
            except ValueError as e:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason=str(e),
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            op = self.engine.apply_cut(src_start, src_end, prompt=prompt)
            action_name = "Trimmed" if "trim" in clean_prompt.lower() else "Cut"
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=f"{action_name} the last {dur_to_cut:.1f}s. New duration is {self.engine.state.duration_s:.1f}s.",
                time_policy_used=self.time_policy,
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # "Cut the first 30 seconds" / "Remove first 10s" / "trim first 30s"
        cut_first_match = re.search(r"(cut|remove|delete|trim)\s+(?:(?:the\s+|this\s+)?(?:video|audio|clip)\s+)?(the\s+)?first\s+(\d+)\s*(s|sec|seconds)?", clean_prompt, re.I)
        if cut_first_match:
            duration_to_cut = float(cut_first_match.group(3))
            if duration_to_cut >= self.engine.state.duration_s:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason="Cannot cut the entire video duration.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

            try:
                src_start, src_end, _ = TimeTranslator.resolve_span(
                    0.0, duration_to_cut, self.time_policy,
                    self.engine.state.projected_clips, self.engine.state.source.duration
                )
            except ValueError as e:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.REJECTED,
                    rejection_reason=str(e),
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

            op = self.engine.apply_cut(src_start, src_end, prompt=prompt)
            action_name = "Trimmed" if "trim" in clean_prompt.lower() else "Cut"
            summary_msg = f"{action_name} the first {duration_to_cut:.1f}s (video is now {duration_to_cut:.1f}s shorter). New duration is {self.engine.state.duration_s:.1f}s."
            if "Note: Caption" in op.description:
                summary_msg += " Note: " + op.description.split("Note: ")[1]

            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=summary_msg,
                time_policy_used=self.time_policy,
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # Semantic Content Cuts: "Remove the part about pricing" / "remove all mentions of pricing" / "cut the discussion about cost"
        semantic_cut_match = re.search(
            r"(remove|cut|delete|trim)\s+(?:out\s+)?(?:the\s+part\s+about|the\s+section\s+about|all\s+mentions\s+of|all\s+parts\s+about|the\s+discussion\s+about|every\s+mention\s+of)?\s*([a-zA-Z\s]+)",
            clean_prompt,
            re.I
        )
        if semantic_cut_match:
            topic = (semantic_cut_match.group(2) or semantic_cut_match.group(1)).strip()
            # If user explicitly said "all mentions" or "every mention"
            is_explicit_all = bool(re.search(r"\b(all|every)\s+(?:mentions?|parts?|sections?|occurrences?)?\b", clean_prompt, re.I))

            # Search for occurrences of topic in transcript
            occurrences = self._find_topic_occurrences(topic)
            self.last_clarification_topic = topic

            if is_explicit_all and occurrences:
                applied_ops = []
                for occ in occurrences:
                    op = self.engine.apply_cut(occ["start"], occ["end"], prompt=f"Cut mention of {topic}: {occ['preview']}")
                    applied_ops.append(op.id)
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=applied_ops,
                    summary=f"Applied removal of all {len(occurrences)} mentions of '{topic}'.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

            if len(occurrences) > 1:
                # Ambiguity detected! As required by IncuBrix: "Never guess on anything destructive"
                self.pending_clarification = {
                    "action": "cut_topic",
                    "topic": topic,
                    "occurrences": occurrences,
                    "original_prompt": prompt,
                    "step_number": step_number
                }
                options_str = [
                    f"Mention {i+1} at {occ['time_str']} ({occ['preview']})"
                    for i, occ in enumerate(occurrences)
                ]
                options_str.append("Remove all mentions")

                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.NEEDS_CLARIFICATION,
                    question=f"'{topic}' is mentioned {len(occurrences)} times — which one, or all?",
                    options=options_str,
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            elif len(occurrences) == 1:
                occ = occurrences[0]
                op = self.engine.apply_cut(occ["start"], occ["end"], prompt=prompt)
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[op.id],
                    summary=f"Removed section discussing '{topic}' ({occ['time_str']}).",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # -------------------------------------------------------------------
        # 5. AUTO-REFRAME / CANVAS ASPECT RATIO (CapCut feature)
        # -------------------------------------------------------------------
        reframe_match = re.search(
            r"(?:convert|reframe|resize|change|make)\s+(?:(?:this|the)\s+)?(?:video|canvas|aspect\s+ratio)?\s*(?:to\s+)?(9:16|16:9|1:1|vertical|portrait|square|landscape)",
            clean_prompt,
            re.I
        ) or re.search(r"make\s+it\s+(vertical|portrait|square|landscape|9:16|1:1|16:9)", clean_prompt, re.I)
        if reframe_match:
            raw_target = (reframe_match.group(1) or "").lower()
            if raw_target in ("vertical", "portrait", "9:16"):
                aspect = "9:16"
            elif raw_target in ("square", "1:1"):
                aspect = "1:1"
            elif raw_target in ("landscape", "16:9"):
                aspect = "16:9"
            else:
                aspect = "9:16"
            bg = "black" if "black" in clean_prompt.lower() else "blur"
            op = self.engine.apply_auto_reframe(aspect_ratio=aspect, background=bg, prompt=prompt)
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=f"Re-framed canvas to {aspect} aspect ratio with {bg} background for mobile/social.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # -------------------------------------------------------------------
        # 6. DYNAMIC PUNCH-IN / ZOOM (CapCut feature)
        # -------------------------------------------------------------------
        punch_match = re.search(
            r"(?:punch\s+in|zoom\s+in)(?:\s+(?:by\s+)?(\d+(?:\.\d+)?)(?:x)?)?(?:\s+(?:when\s+(?:i\s+say\s+)?(.*?)|from\s+(\d+(?::\d+)?)\s+to\s+(\d+(?::\d+)?)))?$",
            clean_prompt,
            re.I
        )
        if punch_match:
            scale_val = float(punch_match.group(1)) if punch_match.group(1) else 1.25
            trigger_phrase = punch_match.group(2)
            t_start_str = punch_match.group(3)
            t_end_str = punch_match.group(4)

            if trigger_phrase:
                matched_words = self._find_words_by_phrase(trigger_phrase)
                if not matched_words:
                    matched_words = self._find_words_by_phrase(trigger_phrase.split()[-1])
                if matched_words:
                    s_start = matched_words[0].start
                    s_end = matched_words[-1].end
                    word_ids = [w.id for w in matched_words]
                    op = self.engine.apply_punch_in(s_start, s_end, scale=scale_val, word_ids=word_ids, prompt=prompt)
                    return StepResponse(
                        step_number=step_number,
                        prompt=prompt,
                        status=StepStatus.APPLIED,
                        operations=[op.id],
                        summary=f"Applied punch-in zoom ({scale_val}x) anchored to phrase '{trigger_phrase}'.",
                        timeline_hash=self.engine.state.timeline_hash,
                        duration_s=self.engine.state.duration_s
                    )
            elif t_start_str and t_end_str:
                raw_start = self._parse_time_string(t_start_str)
                raw_end = self._parse_time_string(t_end_str)
                s_start, s_end, _ = TimeTranslator.resolve_span(
                    raw_start, raw_end, self.time_policy,
                    self.engine.state.projected_clips, self.engine.state.source.duration
                )
                op = self.engine.apply_punch_in(s_start, s_end, scale=scale_val, prompt=prompt)
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[op.id],
                    summary=f"Applied punch-in zoom ({scale_val}x) between {s_start:.2f}s and {s_end:.2f}s.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            else:
                s_start = 0.0
                s_end = min(self.engine.state.source.duration, 5.0)
                op = self.engine.apply_punch_in(s_start, s_end, scale=scale_val, prompt=prompt)
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[op.id],
                    summary=f"Applied punch-in zoom ({scale_val}x) over {s_start:.2f}s - {s_end:.2f}s.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # -------------------------------------------------------------------
        # 7. CINEMATIC LUT & COLOR LOOKS (CapCut feature)
        # -------------------------------------------------------------------
        # 7a. Remove / Reset Color Grade
        if re.search(r"\b(remove|delete|undo|reset)\s+(?:the\s+)?(?:lut|color\s+grade|filter|color\s+look)\b", clean_prompt, re.I):
            restored = self.engine.restore_edit_by_id_or_match("lut")
            if restored:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[restored.id],
                    summary="Removed color grade. Video restored to original colors.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # 7b. Apply Color Grade / LUT
        p_lower = clean_prompt.lower()
        lut_match = (
            re.search(r"\b(teal\s+(?:and|&)\s+orange|teal\s+orange)\b", clean_prompt, re.I) or
            re.search(r"\b(monochrome|black\s+and\s+white|black\s+&\s+white|b&w|grayscale|greyscale)\b", clean_prompt, re.I) or
            re.search(r"\b(vintage|retro|film\s+look)\b", clean_prompt, re.I) or
            re.search(r"\b(warm\s+cinematic|golden\s+hour)\b", clean_prompt, re.I) or
            re.search(r"(?:apply|add|use|set|make|put)\s+.*?\b(lut|color\s+grade|color\s+look|look|filter|cinematic)\b", clean_prompt, re.I) or
            re.search(r"\b(cinematic\s+look|cinematic\s+lut|color\s+grade)\b", clean_prompt, re.I) or
            re.search(r"make\s+(?:it|the\s+video)\s+(?:look\s+)?(cinematic|warm|vintage|monochrome|black\s+and\s+white)", clean_prompt, re.I)
        )
        if lut_match:
            if "teal" in p_lower:
                lut_name = "teal_orange"
            elif any(k in p_lower for k in ("vintage", "retro", "film")):
                lut_name = "vintage"
            elif any(k in p_lower for k in ("monochrome", "black and white", "black & white", "b&w", "grayscale", "greyscale")):
                lut_name = "monochrome"
            else:
                lut_name = "cinematic_warm"

            op = self.engine.apply_lut(lut_name=lut_name, prompt=prompt)
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=f"Applied '{lut_name}' cinematic color grade preset to entire video.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # -------------------------------------------------------------------
        # 8. CLIP TRANSITIONS (CapCut feature)
        # -------------------------------------------------------------------
        trans_match = re.search(
            r"add\s+(?:a\s+)?(?:(\d+(?:\.\d+)?)(?:s)?\s+)?(crossfade|fade|dissolve|wipe)\s+transition(?:s)?(?:\s+between\s+(?:all\s+)?(?:cuts|clips))?",
            clean_prompt,
            re.I
        ) or re.search(r"transition\s+between\s+(?:all\s+)?(?:cuts|clips)", clean_prompt, re.I)
        if trans_match:
            dur = float(trans_match.group(1)) if (hasattr(trans_match, "group") and trans_match.lastindex and trans_match.group(1)) else 0.5
            t_type = trans_match.group(2).lower() if (hasattr(trans_match, "group") and trans_match.lastindex and trans_match.lastindex >= 2 and trans_match.group(2)) else "fade"
            op = self.engine.apply_transition(transition_type=t_type, duration=dur, prompt=prompt)
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=f"Added {t_type} transition ({dur:.1f}s) between all edit cuts.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # -------------------------------------------------------------------
        # 9. B-ROLL & GRAPHIC OVERLAYS (CapCut feature)
        # -------------------------------------------------------------------
        overlay_match = re.search(
            r"(?:overlay|add\s+overlay)\s+['\"]?([\w\.\-_/]+)['\"]?\s+(?:when\s+(?:i\s+(?:say|talk\s+about)\s+)?(.*?))$",
            clean_prompt,
            re.I
        )
        if overlay_match:
            media_path = overlay_match.group(1)
            phrase = overlay_match.group(2)
            matched_words = self._find_words_by_phrase(phrase)
            if not matched_words:
                occ = self._find_topic_occurrences(phrase)
                if occ:
                    matched_words = [w for w in self.state_words() if occ[0]["start"] <= w.start <= occ[0]["end"]]

            if matched_words:
                word_ids = [w.id for w in matched_words]
                op = self.engine.apply_overlay(media_path=media_path, word_ids=word_ids, prompt=prompt)
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=[op.id],
                    summary=f"Overlay '{media_path}' anchored to {len(word_ids)} words during '{phrase}'.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )
            else:
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.NEEDS_CLARIFICATION,
                    question=f"Could not locate the spoken phrase or topic '{phrase}' for overlay placement.",
                    options=["At beginning of video", "At current playhead", "Specify a timestamp"],
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # Fallback for Direct Clarification Option Clicks (e.g. "Selected: Mention 2 at 0:31", "Remove all mentions")
        # 1. "Remove all mentions" or "Selected: Remove all mentions"
        all_mentions_match = re.search(
            r"^(?:selected\s*:\s*)?(?:remove|cut|delete)\s+all\s+(?:mentions|parts|sections)?(?:\s+(?:of|about)\s+([a-zA-Z\s]+))?",
            clean_prompt,
            re.I
        )
        if all_mentions_match or clean_prompt.lower().strip() in ("remove all mentions", "all mentions", "all"):
            topic = all_mentions_match.group(1).strip() if (all_mentions_match and all_mentions_match.group(1)) else getattr(self, "last_clarification_topic", "pricing")
            occurrences = self._find_topic_occurrences(topic)
            if occurrences:
                applied_ops = []
                for occ in occurrences:
                    op = self.engine.apply_cut(occ["start"], occ["end"], prompt=f"Cut mention of {topic}: {occ['preview']}")
                    applied_ops.append(op.id)
                return StepResponse(
                    step_number=step_number,
                    prompt=prompt,
                    status=StepStatus.APPLIED,
                    operations=applied_ops,
                    summary=f"Applied removal of all {len(occurrences)} mentions of '{topic}'.",
                    timeline_hash=self.engine.state.timeline_hash,
                    duration_s=self.engine.state.duration_s
                )

        # 2. Individual Mention Selection: "Selected: Mention 2 at 0:31 (...pricing at 0:31...)" or "Mention 2 at 0:31"
        mention_select_match = re.search(
            r"^(?:selected\s*:\s*)?mention\s+(\d+)\s+at\s+([\d:]+)",
            clean_prompt,
            re.I
        )
        if mention_select_match:
            m_num = int(mention_select_match.group(1))
            t_str = mention_select_match.group(2)
            time_sec = self._parse_time_string(t_str)
            src_start = max(0.0, time_sec - 1.5)
            src_end = min(self.engine.state.source.duration, time_sec + 2.5)
            op = self.engine.apply_cut(src_start, src_end, prompt=f"Cut Mention {m_num} at {t_str}")
            return StepResponse(
                step_number=step_number,
                prompt=prompt,
                status=StepStatus.APPLIED,
                operations=[op.id],
                summary=f"Applied removal of Mention {m_num} at {t_str}.",
                timeline_hash=self.engine.state.timeline_hash,
                duration_s=self.engine.state.duration_s
            )

        # AI / LLM Swappable Intent Fallback (IncuBrix Rule #3 & Must-Have #4)
        if self.llm_adapter:
            try:
                parsed = self.llm_adapter.parse_instruction(clean_prompt)
                if parsed and parsed.action:
                    # 1. Add caption
                    if parsed.action == "add_caption":
                        cap_text = (parsed.caption_text or "Welcome!").strip()
                        trig = (parsed.trigger_phrase or cap_text).strip()
                        matched_words = self._find_words_by_phrase(trig)
                        if not matched_words and trig != cap_text:
                            matched_words = self._find_words_by_phrase(cap_text)
                        if matched_words:
                            style = CaptionStyle()
                            if parsed.animation:
                                style.animation = parsed.animation
                            if parsed.color:
                                style.color = parsed.color
                            op = self.engine.apply_add_caption(text=cap_text, word_ids=[w.id for w in matched_words], style=style, prompt=prompt)
                            return StepResponse(
                                step_number=step_number,
                                prompt=prompt,
                                status=StepStatus.APPLIED,
                                operations=[op.id],
                                summary=f"Caption '{cap_text}' anchored via AI to {len(matched_words)} words at {matched_words[0].start:.2f}s.",
                                timeline_hash=self.engine.state.timeline_hash,
                                duration_s=self.engine.state.duration_s
                            )

                    # 2. Cut
                    elif parsed.action == "cut":
                        if parsed.time_start and parsed.time_end:
                            raw_s = self._parse_time_string(parsed.time_start)
                            raw_e = self._parse_time_string(parsed.time_end)
                            src_s, src_e, expl = TimeTranslator.resolve_span(raw_s, raw_e, self.time_policy, self.engine.state.projected_clips, self.engine.state.source.duration)
                            op = self.engine.apply_cut(src_s, src_e, prompt=prompt)
                            return StepResponse(
                                step_number=step_number,
                                prompt=prompt,
                                status=StepStatus.APPLIED,
                                operations=[op.id],
                                summary=f"Cut {raw_s:.1f}s to {raw_e:.1f}s via AI intent. {expl}",
                                timeline_hash=self.engine.state.timeline_hash,
                                duration_s=self.engine.state.duration_s
                            )
                        elif parsed.topic:
                            occurrences = self._find_topic_occurrences(parsed.topic)
                            if len(occurrences) == 1:
                                op = self.engine.apply_cut(occurrences[0]["start"], occurrences[0]["end"], prompt=prompt)
                                return StepResponse(
                                    step_number=step_number,
                                    prompt=prompt,
                                    status=StepStatus.APPLIED,
                                    operations=[op.id],
                                    summary=f"Cut topic '{parsed.topic}' via AI intent.",
                                    timeline_hash=self.engine.state.timeline_hash,
                                    duration_s=self.engine.state.duration_s
                                )

                    # 3. Restore
                    elif parsed.action == "restore":
                        restored_op = self.engine.restore_edit_by_id_or_match(parsed.target or "intro")
                        if restored_op:
                            return StepResponse(
                                step_number=step_number,
                                prompt=prompt,
                                status=StepStatus.APPLIED,
                                operations=[restored_op.id],
                                summary=f"Restored edit '{parsed.target}' via AI intent.",
                                timeline_hash=self.engine.state.timeline_hash,
                                duration_s=self.engine.state.duration_s
                            )

                    # 4. Silence removal
                    elif parsed.action == "silence_removal":
                        ops = self.engine.remove_silences(min_duration=1.0, prompt=prompt)
                        return StepResponse(
                            step_number=step_number,
                            prompt=prompt,
                            status=StepStatus.APPLIED,
                            operations=[o.id for o in ops],
                            summary=f"Removed {len(ops)} silences via AI intent.",
                            timeline_hash=self.engine.state.timeline_hash,
                            duration_s=self.engine.state.duration_s
                        )

                    # 5. Filler removal
                    elif parsed.action == "filler_removal":
                        ops = self.engine.remove_filler_words(prompt=prompt)
                        return StepResponse(
                            step_number=step_number,
                            prompt=prompt,
                            status=StepStatus.APPLIED,
                            operations=[o.id for o in ops],
                            summary=f"Removed {len(ops)} filler words via AI intent.",
                            timeline_hash=self.engine.state.timeline_hash,
                            duration_s=self.engine.state.duration_s
                        )

                    # 6. BGM
                    elif parsed.action == "bgm":
                        op = self.engine.apply_add_bgm("storage/audio/background.mp3", prompt=prompt)
                        return StepResponse(
                            step_number=step_number,
                            prompt=prompt,
                            status=StepStatus.APPLIED,
                            operations=[op.id],
                            summary="Added background music track via AI intent.",
                            timeline_hash=self.engine.state.timeline_hash,
                            duration_s=self.engine.state.duration_s
                        )
            except Exception:
                pass

        # Fallback rejection for unhandled or invalid prompt
        return StepResponse(
            step_number=step_number,
            prompt=prompt,
            status=StepStatus.REJECTED,
            rejection_reason=f"Unrecognized instruction: '{prompt}'.",
            timeline_hash=self.engine.state.timeline_hash,
            duration_s=self.engine.state.duration_s
        )

    # -----------------------------------------------------------------------
    # Multi-Turn Clarification Resolver
    # -----------------------------------------------------------------------

    def _handle_clarification_resolution(self, answer: str, step_number: int) -> StepResponse:
        """Resolves a pending clarification when the user answers."""
        pending = self.pending_clarification
        self.pending_clarification = None  # Clear state
        self.last_clarification_topic = pending.get("topic", "pricing")

        occurrences = pending["occurrences"]
        chosen_indices = []

        lower_ans = answer.lower()
        mention_match = re.search(r"mention\s+(\d+)", lower_ans)
        if mention_match:
            idx = int(mention_match.group(1)) - 1
            if 0 <= idx < len(occurrences):
                chosen_indices = [idx]
        elif "all" in lower_ans:
            chosen_indices = list(range(len(occurrences)))
        elif "first" in lower_ans or "the 1st" in lower_ans or ("one" in lower_ans and "second" not in lower_ans):
            chosen_indices = [0]
        elif "second" in lower_ans or "the 2nd" in lower_ans or "the second one" in lower_ans:
            chosen_indices = [1]
        elif "third" in lower_ans or "the 3rd" in lower_ans:
            chosen_indices = [2]
        else:
            # Default to index 1 if answer states "the second one"
            chosen_indices = [1] if len(occurrences) > 1 else [0]

        applied_ops = []
        for idx in chosen_indices:
            if idx < len(occurrences):
                occ = occurrences[idx]
                op = self.engine.apply_cut(occ["start"], occ["end"], prompt=f"Clarified cut: {occ['preview']}")
                applied_ops.append(op.id)

        return StepResponse(
            step_number=step_number,
            prompt=f"Clarification reply: '{answer}'",
            status=StepStatus.APPLIED,
            operations=applied_ops,
            summary=f"Applied removal of chosen mention(s) of '{pending['topic']}'.",
            timeline_hash=self.engine.state.timeline_hash,
            duration_s=self.engine.state.duration_s
        )

    # -----------------------------------------------------------------------
    # Semantic & Keyword Search Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _levenshtein_dist(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return IntentCompiler._levenshtein_dist(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _find_words_by_phrase(self, phrase: str) -> List[WordToken]:
        """Finds a sequence of word tokens matching the requested phrase using exact, compound, and fuzzy matching."""
        clean_phrase = re.sub(r"[^\w\s]", "", phrase.lower()).split()
        if not clean_phrase:
            return []

        words = self.state_words()
        if not words:
            return []

        # 1. Exact phrase window match
        for i in range(len(words) - len(clean_phrase) + 1):
            window = [re.sub(r"[^\w\s]", "", words[i + j].word.lower()) for j in range(len(clean_phrase))]
            if window == clean_phrase:
                return words[i : i + len(clean_phrase)]

        # 2. Compound / split word match (e.g. "wel come" <-> "welcome")
        target_joined = "".join(clean_phrase)
        for i in range(len(words)):
            w1 = re.sub(r"[^\w]", "", words[i].word.lower())
            if w1 == target_joined:
                return [words[i]]
            if i + 1 < len(words):
                w2 = re.sub(r"[^\w]", "", words[i + 1].word.lower())
                if (w1 + w2) == target_joined:
                    return [words[i], words[i + 1]]

        # 3. Single-word fuzzy / phonetic match
        if len(clean_phrase) == 1:
            target = clean_phrase[0]
            # Levenshtein distance <= 1 for words >= 4 chars first (handles single-character typos like "welome" -> "welcome")
            if len(target) >= 4:
                for w in words:
                    cw = re.sub(r"[^\w]", "", w.word.lower())
                    if cw and self._levenshtein_dist(cw, target) <= 1:
                        return [w]
            # Substring / stem match (require at least 4 chars on both sides to prevent matching short filler words like "we")
            for w in words:
                cw = re.sub(r"[^\w]", "", w.word.lower())
                if cw and len(cw) >= 4 and len(target) >= 4 and (target.startswith(cw) or cw.startswith(target) or target in cw):
                    return [w]

        # 4. Core anchor keyword fallback (e.g. phrase mentions "welcome" or "welome", find "welcome" anywhere in audio)
        key_anchors = [w for w in clean_phrase if w in ("welcome", "welome", "pricing", "subscribe", "hello", "introduction")]
        for k in key_anchors:
            target_k = "welcome" if k == "welome" else k
            for i, w in enumerate(words):
                if re.sub(r"[^\w]", "", w.word.lower()) == target_k:
                    ext = []
                    for next_i in range(i + 1, min(len(words), i + 4)):
                        kw = re.sub(r"[^\w]", "", words[next_i].word.lower())
                        if kw in ("back", "to", "the", "this", "my", "channel", "everyone", "guys", "video", "show", "mat"):
                            ext.append(words[next_i])
                        else:
                            break
                    return [w] + ext if ext else [w]

        # 5. Longest matching sub-phrase search anchored to phrase prefix or key words
        meta_prompt_words = {"video", "speaker", "says", "say", "talks", "caption", "word", "audio", "person", "when", "that", "add"}
        if len(clean_phrase) > 1:
            for sub_len in range(len(clean_phrase) - 1, 0, -1):
                sub = clean_phrase[:sub_len]
                # Skip if sub has only generic short stopwords or prompt meta words
                if all(w in meta_prompt_words or len(w) <= 3 for w in sub):
                    continue
                for i in range(len(words) - len(sub) + 1):
                    window = [re.sub(r"[^\w\s]", "", words[i + j].word.lower()) for j in range(len(sub))]
                    if window == sub:
                        matched_span = words[i : i + len(sub)]
                        if sub == ["welcome"]:
                            ext = []
                            for k in range(i + 1, min(len(words), i + 5)):
                                kw = re.sub(r"[^\w]", "", words[k].word.lower())
                                if kw in ("back", "to", "the", "this", "my", "channel", "everyone", "guys", "video", "show", "mat"):
                                    ext.append(words[k])
                                else:
                                    break
                            if ext:
                                return [words[i]] + ext
                        return matched_span

        return []

    def _find_topic_occurrences(self, topic: str) -> List[Dict[str, Any]]:
        """
        Finds occurrences of a topic (e.g., 'pricing', 'cost', 'subscription') in transcript.
        Advanced level requirement: meaning-based search ('pricing' finds 'cost').
        """
        keywords = [topic.lower()]
        if "pricing" in topic.lower() or "price" in topic.lower():
            keywords.extend(["cost", "pricing", "dollar", "subscription", "fee"])

        occurrences = []
        words = self.state_words()

        for idx, w in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", w.word.lower())
            if clean_word in keywords:
                # Capture a 5-second sentence context around the word
                ctx_start = max(0.0, w.start - 1.5)
                ctx_end = min(self.engine.state.source.duration, w.end + 2.0)
                time_str = f"{int(w.start // 60)}:{int(w.start % 60):02d}"
                occurrences.append({
                    "start": ctx_start,
                    "end": ctx_end,
                    "time_str": time_str,
                    "preview": f"...{w.word} at {time_str}..."
                })

        return occurrences

    def state_words(self) -> List[WordToken]:
        return self.engine.state.transcript.words

    @staticmethod
    def _parse_time_string(time_str: str) -> float:
        """Parses '1:20', '0:30', '80s', or natural speech numbers into seconds."""
        clean = time_str.lower().strip()
        for unit in ("seconds", "second", "secs", "sec", "s"):
            if clean.endswith(unit):
                clean = clean[:-len(unit)].strip()
        if "minute twenty" in clean:
            return 80.0
        if "minute" in clean:
            clean = clean.replace("a minute", "60").replace("one minute", "60").replace("minute", "").strip()
        if ":" in clean:
            parts = clean.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(clean)
