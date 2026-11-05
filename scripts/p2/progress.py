"""Tiny progress helpers: tqdm if installed, else a plain fallback. Bars only on a TTY (or PALOMARES_TTY=1)."""
import os, sys, time

_TTY = sys.stdout.isatty() or os.environ.get("PALOMARES_TTY") == "1"
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def bar(it, desc="", total=None):
    if total is None:
        try:
            total = len(it)
        except TypeError:
            total = None
    if _tqdm is not None and _TTY:
        return _tqdm(it, desc="  " + desc, total=total, leave=False, ncols=80, file=sys.stdout)
    return _plain(it, desc, total)


def _plain(it, desc, total):
    t0 = time.time(); n = 0
    for x in it:
        yield x
        n += 1
        if _TTY and total:
            w = 28; f = n / total
            sys.stdout.write(f"\r  {desc:24s} [{'#' * int(w * f):{w}s}] {n}/{total} {time.time() - t0:5.1f}s")
            sys.stdout.flush()
    if _TTY and total:
        sys.stdout.write("\r" + " " * 78 + "\r"); sys.stdout.flush()


class Section:
    """`with Section("terrain"):` prints '  - terrain ... 3.2s'"""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time(); return self

    def __exit__(self, *a):
        print(f"  - {self.name:28s} {time.time() - self.t0:6.1f}s", flush=True)
