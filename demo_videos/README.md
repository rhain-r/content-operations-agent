# demo_videos

Put the videos you want on the public showcase here, then run:

```bash
uv run python -m video_qa.export --videos demo_videos/ --out docs/
```

The raw files in this folder are not committed (see `.gitignore`); only the web copies the
export writes to `docs/` are published. Everything in `docs/` is visible to anyone on the
internet, so only add videos you're happy to make public.
