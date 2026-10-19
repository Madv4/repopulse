"""Small Phase 1 CLI entry point."""

import argparse

from repopulse.config import Settings
from repopulse.llm.factory import create_provider


def main() -> None:
    """Run a local configuration check without making network calls."""
    parser = argparse.ArgumentParser(prog="repopulse")
    parser.add_argument("command", choices=["doctor"])
    args = parser.parse_args()
    if args.command == "doctor":
        settings = Settings()
        provider = create_provider(settings)
        print(f"RepoPulse Phase 1 ready; LLM provider: {provider.name}")
        github_status = (
            "GitHub authentication: configured"
            if settings.github_token
            else "GitHub: public access"
        )
        print(github_status)


if __name__ == "__main__":
    main()
