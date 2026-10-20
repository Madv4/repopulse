"""Human-written, URL-labeled paraphrases for the pinned large Click snapshot."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "fixtures/click_large_snapshot.json"
OUT = ROOT / "datasets/retrieval_large.jsonl"

# Target issue number, paraphrased query, optional similar but different issue number.
# Labels were written before running retrieval variants. The distractors are real
# recorded Click issues whose nearby wording or topic can confuse retrieval.
LABELS: list[tuple[int, str, int | None]] = [
    (3362, "Where does the usage formatter split a long flag at a dash?", 3016),
    (3887, "Which report says applying blue styling to precolored shell output fails?", 2985),
    (3886, "Which pager report says an interrupt while following output exits the program?", 350),
    (2645, "Where can opening a lazy input stream too early break a FIFO?", 2429),
    (3885, "Which report says tab completion unexpectedly runs parameter callbacks?", 2184),
    (3016, "Where was better line layout in help output requested?", 3362),
    (3882, "Who wanted command help to be sent to Unix manual pages?", None),
    (3822, "Which issue asks for a typed path parameter preserving path_type?", None),
    (2847, "Where does completion fail for a long option with an attached equals value?", 3135),
    (3832, "Which issue warns that terminal control bytes in errors can alter display?", None),
    (1411, "Where was custom prompting by a parameter type proposed?", 2795),
    (3831, "Which CI report says stress jobs actually execute ordinary tests?", None),
    (3802, "Where can pressing Ctrl C during prompt race with command shutdown?", 3886),
    (3135, "Which request concerns partial matches when completing shell paths?", 173),
    (173, "Where was interactive prompt tab completion first requested?", 3135),
    (3840, "Which Windows editor report cites WinError 14001 or 87?", 2582),
    (2582, "Which issue says opening Visual Studio Code through edit fails on Windows?", 3840),
    (3010, "Where can a help message not retain one intentional line break?", 3016),
    (3036, "Which request asks for native set and frozenset argument conversion?", None),
    (3788, "Where was an invisible positional argument decorator requested?", None),
    (2121, "Which Windows help report crashes when non-ASCII characters appear?", None),
    (3766, "Which contributing guide points to a deleted Flask workflow?", None),
    (2943, "Where does pipepager fail with a multi-call executable?", 3416),
    (2795, "Who wanted a navigable menu of choices at a prompt?", 1411),
    (3571, "Which progress display fails to reach full completion with show_pos?", 3167),
    (2429, "Where does delayed File opening fail specifically on Windows?", 2645),
    (2992, "Which request wants usage text included in command metadata?", None),
    (3731, "Which Windows pager bug hides a text-to-binary TypeError behind WinError 32?", 3732),
    (3732, "Which report says echo_via_pager regressed on Windows after version 8.4?", 3731),
    (2184, "Where are previously parsed arguments missing from completion callback context?", 3885),
    (3416, "Which pager bug treats any r in LESS arguments as raw mode?", 2943),
    (2154, "Where does launching via xdg-open leave a process behind in WSL2?", None),
    (2676, "Which feature asks to parse minus-prefixed integers as values?", None),
    (3662, "Which Windows prompt bug writes its tail and test reply to stdout?", None),
    (3547, "Where can a clamped float range unexpectedly return an integer?", 3133),
    (3487, "Which issue says echo rejects empty byte strings?", None),
    (3336, "Where is an optional value pattern broken with is_flag disabled?", 3085),
    (3193, "Which issue reports that a Sentinel cannot be unpickled in another process?", None),
    (3176, "Who requested completion descriptions sized to terminal width?", None),
    (3167, "Which request adds OSC progress reporting escape sequences?", 3571),
    (3133, "Where should FloatRange reject a minimum greater than its maximum?", 3547),
    (3132, "Which batch helper loses values when its requested group size exceeds input?", 3131),
    (3100, "Where are flags after positional arguments parsed incorrectly in a group?", None),
    (3085, "Which issue says optional values stopped working altogether?", 3084),
    (3071, "Where does a feature flag's true default depend on declaration order?", 3113),
    (3039, "Which pager request says subprocess should avoid a shell after argv is split?", None),
    (3028, "Where does a StrEnum prompt show values but demand names?", 2970),
    (3000, "Which group command named list causes a false extra-argument error?", None),
    (2985, "Which issue reports that bold text unexpectedly changes hue?", 3887),
    (2959, "Where was matching enumeration values during token normalization requested?", 2970),
    (2939, "Which runner report exits unsuccessfully when File reads stdin?", None),
    (2916, "Where is a command-name suffix silently removed from a function?", 2932),
    (2906, "Which regression breaks nested-group command completion?", None),
    (2813, "Where is the last option mishandled by the incomplete-option loop?", None),
    (2786, "Which issue says a multi-value option overwrites a callback result?", None),
]


def build() -> list[dict[str, object]]:
    """Resolve each reviewed issue number to its recorded canonical URL."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    by_number = {row["number"]: row for row in snapshot["issues"]}
    cases: list[dict[str, object]] = []
    for index, (number, query, distractor) in enumerate(LABELS, 1):
        target = by_number[number]
        if target.get("pull_request"):
            raise ValueError(f"Target {number} is not an issue")
        if distractor is not None and by_number[distractor].get("pull_request"):
            raise ValueError(f"Distractor {distractor} is not an issue")
        cases.append(
            {
                "id": f"click-large-{index:02}",
                "query": query,
                "relevant_urls": [target["html_url"]],
                "distractor_urls": [by_number[distractor]["html_url"]] if distractor else [],
                "paraphrased": True,
            }
        )
    OUT.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    return cases


if __name__ == "__main__":
    print(f"Wrote {len(build())} labeled retrieval cases to {OUT}")
