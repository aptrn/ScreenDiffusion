"""`python -m bench` - see bench/cli.py."""

import sys

from bench.cli import main


def _survive_unprintable_characters() -> None:
    """Never let a status line be what ends a run.

    `wrapper.py` prints emoji from `set_t_index_list`, and a Windows console is
    cp1252: printing a check mark there raises `UnicodeEncodeError` out of the
    middle of a measurement, which is how a 48-frame run dies after loading a 5 GB
    engine. Replacing the character costs nothing; the encoding itself is the
    console's business and is left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass


if __name__ == "__main__":
    _survive_unprintable_characters()
    sys.exit(main())
