"""
Unified Test Runner (Integration Standard #1).

Runs unit tests, integration tests, and the end-to-end 8-step worked example session.
Requires zero external API keys and runs completely offline.
"""

import sys
import pytest

if __name__ == "__main__":
    print("=" * 60)
    print("Running IncuBrix Test Suite (Offline / Zero API Keys)")
    print("=" * 60)

    # Run pytest on tests directory
    exit_code = pytest.main([
        "-v",
        "--tb=short",
        "tests/"
    ])

    if exit_code == 0:
        print("\n[SUCCESS] All unit, integration, and E2E tests PASSED!")
    else:
        print(f"\n[FAILURE] Tests failed with exit code: {exit_code}")

    sys.exit(exit_code)
