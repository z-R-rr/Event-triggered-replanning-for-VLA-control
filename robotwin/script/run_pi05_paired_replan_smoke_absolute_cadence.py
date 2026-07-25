"""Run the 3+3 paired smoke with natural replans fixed at r0 multiples."""

import sys

from run_pi05_paired_replan_smoke import main


if __name__ == "__main__":
    if "--absolute-r0-cadence" not in sys.argv:
        sys.argv.append("--absolute-r0-cadence")
    main()
