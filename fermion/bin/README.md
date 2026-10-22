# fermion/bin

Official PyPI wheels of `fermion-research` bundle one prebuilt native
binary here: the `.tv4z` transport decoder for Apple Silicon
(`tv4z-macos-arm64`, with its `.sha256` pin alongside). It decodes a
downloaded `.tv4z` transport archive into the runnable `.bin` container
locally, which shrinks the 8B download from 3.88 GB to 2.56 GB.

The decoder only runs on Apple Silicon. On every other platform it is
inert: the loader falls back to downloading the raw `.bin` container
directly, and everything still works. The CLI prints which path is active
(`fermion info`).

License: Apache-2.0 (flagship is a Qwen3-8B derivative, Apache-2.0 upstream).
