"""Tools to aid showing results."""

import json
import subprocess
import sys
import time
from functools import wraps
from pathlib import Path

import colorama as colr

colr.init()  # for Windows
# warn = colr.Back.YELLOW + "Warning:" + colr.Style.RESET_ALL
warn = "Warning:"


# Running in iPython?
ip = __import__("IPython").get_ipython() if "IPython" in sys.modules else None


def confirm_cold_call(script: str, seconds: int = 300):
    """Run decorated function only if it was last run within `seconds`, or by user confirmation."""

    def decorator(func):
        @wraps(func)
        def wrapper():
            # Already cancelled in this ipytho session ⇒ re-cancel
            if ip:
                fkey = (func.__name__, func.__module__)
                if confirm_cold_call.register.get(fkey, None) == ip.execution_count:
                    print(f"Re-ignoring invocation of {func.__name__}.")
                    return

            # ╔═══════════╗
            # ║ timestamp ║
            # ╚═══════════╝
            script_path = Path(script)
            timestamp_file = script_path.parent / ".call_timestamps"

            # write
            def update_timestamp():
                timestamps[str(script_path)] = time.time()
                with open(timestamp_file, "w") as f:
                    json.dump(timestamps, f)

            # read
            if timestamp_file.exists():
                with open(timestamp_file) as f:
                    timestamps = json.load(f)
            else:
                timestamps = {}

            # check
            now = time.time()
            need_confirmation = True
            last_run = timestamps.get(str(script_path))
            if last_run is not None and now - last_run <= seconds:
                need_confirmation = False

            def cancel():
                print("Operation cancelled.")
                if ip:
                    fkey = (func.__name__, func.__module__)
                    confirm_cold_call.register[fkey] = ip.execution_count

            def call():
                func()
                update_timestamp()

            # ╔══════════════════════╗
            # ║ ask for confirmation ║
            # ╚══════════════════════╝
            try:
                if need_confirmation:
                    print(
                        f"It's been more than {seconds // 60}m since confirmed invocation."
                        f" You sure you want to {func.__name__}?"
                    )
                    try:
                        if input("Confirm [y/N]: ").strip().lower() == "y":
                            call()
                        else:
                            cancel()
                    except KeyboardInterrupt:
                        print()  # To move to a new line after Ctrl-C
                        cancel()
                else:
                    call()
            finally:
                pass

        wrapper()

        # return wrapper
        return lambda *a, **b: print("Function already called (in decorator).")

    return decorator


confirm_cold_call.register = {}


def stripe(rows, start=0, width=2):
    """Apply 'shading' to alternate lines in `rows` (str)."""
    if not isinstance(rows, list):
        rows = rows.splitlines()
    for i in range(start, len(rows)):
        if ((i - start) // width) % 2 == 0:
            rows[i] = colr.Fore.BLACK + colr.Back.LIGHTWHITE_EX + rows[i] + colr.Style.RESET_ALL
    rows = "\n".join(rows)
    return rows


def yank(txt, append=False):
    "Copy to clipboard (mac/darwin)."

    if append:
        old = subprocess.check_output("pbpaste", env={"LANG": "en_US.UTF-8"}).decode("utf-8")
        txt = old + "\n" + txt

    process = subprocess.Popen("pbcopy", env={"LANG": "en_US.UTF-8"}, stdin=subprocess.PIPE)
    process.communicate(txt.encode("utf-8"))
