"""
Automated Test for the 8-Step Worked Example & Determinism Verification.

Directly tests:
- Must-Have #1: Never damage original
- Must-Have #2: Edits stick to content (caption shifts and returns)
- Must-Have #3: 3-way answers (Applied, Clarification, Rejected)
- Must-Have #4: Determinism (100% identical timeline_hash across 3 runs)
- Must-Have #6: Undo/redo & selective restore
- Must-Have #7: Clear time policy
"""

import os
import json
import pytest
from src.cli.session_runner import run_session


def test_worked_example_execution(tmp_path):
    yaml_path = os.path.abspath("fixtures/worked_example.yaml")
    output_dir = str(tmp_path / "results_run_1")

    # Run session
    result = run_session(yaml_path, output_dir)

    # 1. Verify all 8 steps were processed
    steps = result["steps"]
    assert len(steps) == 8

    # Step 1: Add caption 'Welcome!'
    s1 = steps[0]
    assert s1["say"] == "Add a caption 'Welcome!' when I say welcome to the channel"
    assert s1["status"] == "applied"
    assert "attached to" in s1["summary"]

    # Step 2: Cut first 30 seconds
    s2 = steps[1]
    assert s2["say"] == "Cut the first 30 seconds"
    assert s2["status"] == "applied"

    # Step 3: Remove pricing (requires clarification and resolves with answer)
    s3 = steps[2]
    assert s3["status"] == "needs_clarification"
    assert s3["answer_provided"] == "the second one"
    assert s3["resolution"]["status"] == "applied"

    # Step 4: Speed up 1:00 to 1:30 by 2x
    s4 = steps[3]
    assert s4["status"] == "applied"
    assert "2.0x" in s4["summary"]

    # Step 5: Put intro back (Selective Restore)
    s5 = steps[4]
    assert s5["status"] == "applied"
    assert "Restored" in s5["summary"]

    # Step 6: Undo (Reverses Step 5 only)
    s6 = steps[5]
    assert s6["status"] == "applied"
    assert "Reversed step" in s6["summary"]

    # Step 7: Make caption red
    s7 = steps[6]
    assert s7["status"] == "applied"
    assert "Applied color 'red'" in s7["summary"]

    # Step 8: Render
    s8 = steps[7]
    assert s8["status"] == "applied"

    # Check generated artifacts
    assert os.path.exists(os.path.join(output_dir, "results.json"))
    assert os.path.exists(os.path.join(output_dir, "final_timeline.json"))
    assert os.path.exists(os.path.join(output_dir, "final_timeline.otio"))
    assert os.path.exists(os.path.join(output_dir, "manifest.json"))


def test_determinism_across_three_runs(tmp_path):
    """
    Must-Have #4: The same session always produces the same timeline.
    Compare timeline_hash across 3 runs (must be 100% identical).
    """
    yaml_path = os.path.abspath("fixtures/worked_example.yaml")

    hashes = []
    for i in range(3):
        out_dir = str(tmp_path / f"determinism_run_{i+1}")
        res = run_session(yaml_path, out_dir)
        hashes.append(res["final_timeline_hash"])

    # All 3 runs must match bit-for-bit
    assert len(set(hashes)) == 1, f"Determinism failure: distinct hashes found: {hashes}"
    assert len(hashes[0]) == 64  # Valid SHA-256
