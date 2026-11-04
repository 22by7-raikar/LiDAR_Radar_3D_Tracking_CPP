#!/usr/bin/env python3
"""Assembles the PNG frames saved by `fusion_viewer --record DIR` into a GIF.

    python3 scripts/frames_to_gif.py DIR out.gif [--width 760] [--fps 12]

Requires pillow.
"""

import argparse
import glob
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")))
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        h = round(im.height * args.width / im.width)
        frames.append(im.resize((args.width, h), Image.LANCZOS)
                      .convert("P", palette=Image.ADAPTIVE, colors=128))
    frames[0].save(args.out, save_all=True, append_images=frames[1:], loop=0,
                   duration=round(1000 / args.fps), optimize=True)
    print("wrote", os.path.basename(args.out), f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
