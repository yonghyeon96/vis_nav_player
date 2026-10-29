"""Our agent: drive cell by cell to the goal found in the exploration data.

    uv run source/my_agent.py                      # the challenge in .env (VIS_NAV_CHALLENGE)
    uv run source/my_agent.py --challenge <id>     # one run on the server
    uv run source/my_agent.py --local 12           # local maze 12, no attempt, no key

Before connecting it rebuilds the explorer's path in cell coordinates from the exploration
images, finds where the goal photos were taken, and plans a route over the cells the
explorer drove. Driving pins the pose to the wall lattice in every frame; the last few
centimetres are steered by the goal photos themselves. How and why: cell_drive/README.md
and REPORT_NOTES_KO.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # cell_drive/ lives next to source/

import cli  # noqa: E402
from vis_nav_sdk import run  # noqa: E402

from cell_drive.agent import CellAgent  # noqa: E402


class MyAgent(CellAgent):
    """The submitted agent (cell_drive.agent.CellAgent)."""


def main():
    p = cli.parser(__doc__)
    p.add_argument("--data", default=None, help="exploration data directory (default: download)")
    p.add_argument("--capture", type=int, default=None, help="ticks per recorded frame")
    args = p.parse_args()
    data_dir = cli.exploration_data(args, args.data)
    result = run(MyAgent(data_dir, args.capture), cli.challenge(args), **cli.run_options(args))
    print(result)


if __name__ == "__main__":
    main()
