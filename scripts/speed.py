#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28.1", "numpy>=2.5.4", "soundfile>=0.14.0", "tqdm>=4.67.1"]
# ///
"""wavo against other engines on this Mac: same model, same audio, one engine at a time.

    scripts/speed.py download
    scripts/speed.py run [--engines wavo-gpu,sherpa-onnx|all] [--models parakeet-v3,gigaam-v3|all]
                         [--cooldown 30] [--clean]

`download` is optional: it caches the audio, a public-domain LibriVox recording of about an hour per
language in `.cache/data/`, cut once at pauses into 2-15 s pieces and clips of 1, 2, 5, 10 and 30 s
in `.cache/speed/`. `run` does that too when it is missing, fetches and builds each engine into
`.cache/engines/` (cmake, ninja and the Python engines through uv, nothing on the system), then
measures, keeps each result in `results/speed-<chip>.json` as it comes and continues where it
stopped. A row whose engine changed version starts over; `--clean` starts the chosen rows over. An
engine that fails stays in its row with the reason, and the others go on.

Parakeet TDT v3 runs on English, GigaAM v3 e2e-rnnt on Russian. Every engine is one process, one
model load and a list of WAV files, printing JSON lines shaped like `transcribe-cli --batch
--batch-jsonl`. Engines differ in quantization: each row says its version, backend and precision.

- `latency`: each clip listed 6 times in a process of its own: load time, the first call and the
  median of the other 5.
- `hour`: the pieces of the recording in one process, 3 rounds with the engines alternating inside a
  round, `--cooldown` seconds between runs: wall time, seconds of audio per second of wall time,
  peak memory and words. The row has the medians and all three runs.
"""

import argparse
import atexit
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from functools import cache
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median

import httpx
import numpy as np
import soundfile as sf
from tqdm import tqdm

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
DATA = REPO / ".cache" / "data"
WORK = REPO / ".cache" / "speed"
ENGINES_DIR = REPO / ".cache" / "engines"
WAVO = REPO / "target" / "release" / "wavo"
BATCH = REPO / "target" / "release" / "examples" / "batch"
TIME = "/usr/bin/time"
RUNS = 3
CALLS = 6
SECONDS = [1, 2, 5, 10, 30]
PIECE = (2, 15)
RATE = 16000
PORT = 8124
BAR = {"ncols": 80, "leave": False}
BYTES = BAR | {
    "unit": "B",
    "unit_scale": True,
    "bar_format": "{desc} |{bar}| {n_fmt:>6} / {total_fmt:>6} {rate_fmt:>10}",
}
FILES = BAR | {"unit": "file", "bar_format": "{desc} |{bar}| {n:>4} / {total:>4} {rate_fmt:>10}"}
STEPS = BAR | {"bar_format": "{desc} |{bar}| {n:>4} / {total:>4}"}

ARCHIVE = "https://archive.org/download"
HOUR = {
    "en": {
        "url": f"{ARCHIVE}/memoirsofsherlockholmesv3_1502_librivox/memoirsofsherlockholmes_01_doyle_64kb.mp3",
        "sha256": "4acb5f44d312eb60cc6d34e5f95162b2cdb9239f49adc6878a4874dc803d070a",
    },
    # Записки охотника, Гамлет Щигровского уезда.
    "ru": {
        "url": f"{ARCHIVE}/zapiskiohotnika_2409_librivox/zapiskiohotnika_20_turgenev_64kb.mp3",
        "sha256": "e5e76fba9a24b776c4ae1286c6b76e6c3fb011066a511d665d67f5d4a8f47536",
    },
}
LANGUAGE = {"parakeet-v3": "en", "gigaam-v3": "ru"}

TRANSCRIBE_TAG = "v0.4.1"
TRANSCRIBE_URL = "https://github.com/handy-computer/transcribe.cpp"
TRANSCRIBE_CLI = ENGINES_DIR / "transcribe.cpp" / "build" / "bin" / "transcribe-cli"

PARAKEET_TAG = "v0.6.1"
PARAKEET_URL = "https://github.com/mudler/parakeet.cpp"
PARAKEET_SERVER = ENGINES_DIR / "parakeet.cpp" / "build" / "examples" / "server" / "parakeet-server"
PARAKEET_REV = "07a08a3a405d20c5434b9a1442f6e6cf571f57ad"
PARAKEET_GGUF = {
    "url": f"https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/{PARAKEET_REV}/tdt-0.6b-v3-q8_0.gguf",
    "sha256": "4d69a4a6683f4f2d952bad794c1357ca6eb628027695b4699c5a9ad4cd07d757",
}
PARAKEET_MODEL = ENGINES_DIR / "parakeet.cpp" / "tdt-0.6b-v3-q8_0.gguf"

SHERPA_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
SHERPA = {
    "parakeet-v3": {
        "name": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        "sha256": "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
    },
    # `punct` is the e2e-rnnt export, the model wavo's `gigaam-v3` is.
    "gigaam-v3": {
        "name": "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16",
        "sha256": "f9620a0099019c6afcee26525ef9ed3297fa50dd5691c1902af0c948fc1a470b",
    },
}

MLX_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
# Python engines: this file again, through uv, with the package on top of the header's.
PACKAGES = {"parakeet-mlx": "parakeet-mlx>=0.5.3", "sherpa-onnx": "sherpa-onnx>=1.13.8"}

FLUID_TAG = "v0.17.7"
FLUID_DIR = ENGINES_DIR / "fluid-batch"  # not `fluidaudio`: SwiftPM takes the package for its own dependency
FLUID_BIN = FLUID_DIR / ".build" / "release" / "fluid-batch"
FLUID_PACKAGE = f"""// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "fluid-batch",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio", exact: "{FLUID_TAG[1:]}")
    ],
    targets: [
        .executableTarget(
            name: "fluid-batch",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]
        )
    ]
)
"""
# `fluid-batch MODELS_DIR LIST`: the audio is read before each call's clock starts, as in
# examples/batch.rs. The default encoder: int8, on the Neural Engine.
FLUID_BATCH = """import FluidAudio
import Foundation

func emit(_ fields: [String: Any]) throws {
    var line = try JSONSerialization.data(withJSONObject: fields)
    line.append(10)
    FileHandle.standardOutput.write(line)
}

func ms(since start: Date) -> Double {
    Date().timeIntervalSince(start) * 1000
}

@main
struct Batch {
    static func main() async throws {
        let args = CommandLine.arguments
        let folder = AsrModels.defaultCacheDirectory(for: .v3).lastPathComponent
        let models = URL(fileURLWithPath: args[1]).appendingPathComponent(folder)

        let loading = Date()
        let asr = AsrManager()
        try await asr.loadModels(try await AsrModels.downloadAndLoad(to: models, version: .v3))
        try emit(["type": "batch_header", "load_ms": ms(since: loading)])

        let layers = await asr.decoderLayerCount
        let converter = AudioConverter()
        for file in try String(contentsOfFile: args[2], encoding: .utf8).split(separator: "\\n") {
            let samples = try converter.resampleAudioFile(path: String(file))
            var state = TdtDecoderState.make(decoderLayers: layers)
            let start = Date()
            let result = try await asr.transcribe(samples, decoderState: &state)
            try emit(["file": String(file), "text": result.text, "ms": ms(since: start)])
        }
    }
}
"""

ANE_MEMORY = "the weights live in the system's Neural Engine service, outside the peak memory"
ENGINES = {
    "wavo-gpu": {
        "backend": "Metal",
        "precision": {"parakeet-v3": "Q8_0", "gigaam-v3": "Q8_0"},
        "notes": "call: Model::transcribe",
    },
    "wavo-ane": {
        "backend": "Neural Engine",
        "precision": {"parakeet-v3": "int8 encoder", "gigaam-v3": "int8 encoder"},
        "notes": f"call: Model::transcribe; {ANE_MEMORY}",
    },
    "transcribe.cpp": {
        "backend": "Metal",
        "precision": {"parakeet-v3": "Q8_0", "gigaam-v3": "Q8_0"},
        "notes": "call: mel_ms + encode_ms + decode_ms, a few ms below the call's wall time",
    },
    "fluidaudio": {
        "backend": "Neural Engine",
        "precision": {"parakeet-v3": "int8 encoder"},
        "notes": f"call: AsrManager.transcribe, audio padded to 15 s; {ANE_MEMORY}",
    },
    "parakeet-mlx": {
        "backend": "Metal (MLX)",
        "precision": {"parakeet-v3": "bf16"},
        "notes": "call: logmel + generate; load has the imports; wall and peak leave out uv",
    },
    "sherpa-onnx": {
        "backend": "CPU",
        "precision": {"parakeet-v3": "int8", "gigaam-v3": "int8 encoder"},
        "notes": "threads: the performance cores; load has the import; wall and peak leave out uv",
    },
    "parakeet.cpp": {
        "backend": "Metal",
        "precision": {"parakeet-v3": "q8_0"},
        "notes": "call: a parakeet-server request; load: start to /health; wall, peak: the server",
    },
}

# Only for what runs the engines: the person's own environment stays as it is.
ENV = os.environ | {
    "UV_CACHE_DIR": str(ENGINES_DIR / "uv"),
    "UV_PYTHON_INSTALL_DIR": str(ENGINES_DIR / "python"),
    "HF_HOME": str(ENGINES_DIR / "hf"),
}


# MARK: Audio


def hour_file(lang: str) -> Path:
    return DATA / f"hour-{lang}.mp3"


def hour_list(lang: str) -> Path:
    return WORK / lang / "hour.txt"


def clip_list(lang: str, seconds: int) -> Path:
    return WORK / lang / f"{seconds}s.txt"


def fetch(client: httpx.Client, url: str, to: Path, bar: tqdm):
    """`url` into `to`, continuing what an earlier try left there."""
    for attempt in range(5):
        have = to.stat().st_size if to.exists() else 0
        headers = {"Accept-Encoding": "identity", **({"Range": f"bytes={have}-"} if have else {})}
        try:
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 416:  # nothing past `have`: it is all there
                    return

                r.raise_for_status()
                with open(to, "ab" if r.status_code == 206 else "wb") as f:
                    for chunk in r.iter_raw(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))

            return
        except httpx.TransportError:
            if attempt == 4:
                raise

            time.sleep(5)


def size_of(client: httpx.Client, url: str) -> int:
    r = client.head(url, headers={"Accept-Encoding": "identity"})
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def get_file(label: str, url: str, to: Path, sha256: str):
    """`url` into `to`, which takes its name once its hash is right."""
    to.parent.mkdir(parents=True, exist_ok=True)
    part = to.with_suffix(".part")
    have = part.stat().st_size if part.exists() else 0
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        with tqdm(total=size_of(client, url), initial=have, desc=label, **BYTES) as bar:
            fetch(client, url, part, bar)

    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    if digest != sha256:
        part.unlink()
        raise RuntimeError(f"{to.name}: sha256 {digest}, expected {sha256}")

    part.rename(to)


def resample(pcm: np.ndarray, rate: int) -> np.ndarray:
    """`pcm` at 16 kHz, in one FFT over the whole recording."""
    if rate == RATE:
        return pcm

    n = len(pcm) * RATE // rate
    return np.fft.irfft(np.fft.rfft(pcm)[: n // 2 + 1], n) * (n / len(pcm))


def loudness(pcm: np.ndarray) -> np.ndarray:
    """The mean level of each 20 ms of `pcm`."""
    frame = RATE // 50
    return np.abs(pcm[: len(pcm) // frame * frame]).reshape(-1, frame).mean(axis=1)


def cut(pcm: np.ndarray, seconds: int) -> np.ndarray:
    """`seconds` of `pcm`, in 20 ms steps: quiet at both ends, with loud speech between them."""
    frame, n = RATE // 50, seconds * 50
    loud = loudness(pcm)
    sums = np.concatenate([[0], np.cumsum(loud)])
    inside = (sums[n:] - sums[:-n]) / n
    edges = loud[:-n] + loud[n:]
    start = int(np.argmin(edges - inside[: len(edges)])) * frame
    return pcm[start : start + seconds * RATE]


def pieces(pcm: np.ndarray) -> list[np.ndarray]:
    """`pcm` cut at the quietest 200 ms of each 2-15 s stretch; a last bit under 2 s goes."""
    frame, shortest, longest = RATE // 50, PIECE[0] * 50, PIECE[1] * 50
    quiet = np.convolve(loudness(pcm), np.ones(10) / 10, "same")
    parts, start = [], 0
    while len(quiet) - start > longest:
        end = start + shortest + int(np.argmin(quiet[start + shortest : start + longest]))
        parts.append(pcm[start * frame : end * frame])
        start = end

    tail = pcm[start * frame :]
    return parts + [tail] if len(tail) >= PIECE[0] * RATE else parts


def write_audio(lang: str, label: str):
    """A language's pieces, clips and their lists; the list of the pieces comes last."""
    folder = WORK / lang
    (folder / "pieces").mkdir(parents=True, exist_ok=True)
    pcm, rate = sf.read(hour_file(lang), dtype="float64", always_2d=True)
    pcm = resample(pcm.mean(axis=1), rate)
    for seconds in SECONDS:
        clip = folder / f"{seconds}s.wav"
        sf.write(clip, cut(pcm, seconds), RATE, subtype="PCM_16")
        clip_list(lang, seconds).write_text(f"{clip}\n" * CALLS)

    files = []
    for n, part in enumerate(tqdm(pieces(pcm), desc=label, **FILES), 1):
        files.append(folder / "pieces" / f"{n:05}.wav")
        sf.write(files[-1], part, RATE, subtype="PCM_16")

    hour_list(lang).write_text("".join(f"{x}\n" for x in files))
    print(f"{label}  {len(files):>4} pieces")


def get_audio():
    """What is missing of the audio; `run` calls it too, so `download` is only a head start."""
    for n, lang in enumerate(HOUR, 1):
        label = f"{n}/{len(HOUR)} {hour_file(lang).name}"
        if not hour_file(lang).exists():
            get_file(label, HOUR[lang]["url"], hour_file(lang), HOUR[lang]["sha256"])

        if not hour_list(lang).exists():
            write_audio(lang, label)


def download(args):
    get_audio()
    print(f"audio in {DATA}, pieces and clips in {WORK}")


# MARK: Engines


def sh(label: str, cmds: list[list[str]], cwd: Path | None = None):
    """Commands one by one, a step of the bar each; their output goes to a log."""
    log = WORK / "build.log"
    with open(log, "w") as out, tqdm(total=len(cmds), desc=label, **STEPS) as bar:
        for cmd in cmds:
            r = subprocess.run(cmd, stdout=out, stderr=out, cwd=cwd, env=ENV)
            if r.returncode:
                said = log.read_text(errors="replace").strip().splitlines()
                raise RuntimeError(f"{cmd[0]}: {said[-1] if said else r.returncode}")

            bar.update()


def cmake_build(label: str, name: str, url: str, tag: str, target: str, flags: list[str]):
    """A tag of `url` into `.cache/engines/<name>/`, with cmake and ninja from uv."""
    folder = ENGINES_DIR / name
    source, build = folder / "source", folder / "build"
    folder.mkdir(parents=True, exist_ok=True)
    tools = ["uv", "run", "--no-project", "--with", "cmake", "--with", "ninja", "cmake"]
    if not source.exists():
        part = folder / "source.part"
        shutil.rmtree(part, ignore_errors=True)
        clone = ["git", "clone", "--depth", "1", "--branch", tag, "--recurse-submodules"]
        sh(label, [[*clone, url, str(part)]])
        part.rename(source)

    configure = ["-S", str(source), "-B", str(build), "-G", "Ninja"]
    configure += ["-DCMAKE_BUILD_TYPE=Release", *flags]
    sh(label, [[*tools, *configure], [*tools, "--build", str(build), "--target", target]])


@cache
def build_wavo() -> str:
    """wavo's version and commit, once a run: transcribe.cpp needs its `wavo pull` too."""
    cargo = ["cargo", "build", "--release", "--locked", "--bin", "wavo", "--example", "batch"]
    if shutil.which("cargo"):  # a machine without it runs the binaries copied there
        sh("1/1 cargo build", [cargo], cwd=REPO)

    wavo = subprocess.run([str(WAVO), "--version"], capture_output=True, text=True).stdout
    cmd = ["git", "-C", str(REPO), "describe", "--always", "--dirty", "--exclude=*"]
    rev = subprocess.run(cmd, capture_output=True, text=True).stdout
    return f"{wavo.split()[-1]} {rev.strip()}"


@cache
def gguf(model: str) -> str:
    """The model's path; `wavo pull` also prepares it for the Neural Engine."""
    with tqdm(total=1, desc=f"1/1 wavo pull {model}", **STEPS):
        r = subprocess.run([str(WAVO), "pull", model], capture_output=True, text=True)

    if r.returncode:
        raise RuntimeError(r.stderr.strip().splitlines()[-1])

    return r.stdout.strip()


def setup_fluidaudio(label: str):
    if FLUID_BIN.exists():
        return

    sources = FLUID_DIR / "Sources" / "fluid-batch"
    sources.mkdir(parents=True, exist_ok=True)
    (FLUID_DIR / "Package.swift").write_text(FLUID_PACKAGE)
    (sources / "Batch.swift").write_text(FLUID_BATCH)
    local = [f"--{x}-path={FLUID_DIR / 'swiftpm' / x}" for x in ("cache", "config", "security")]
    sh(label, [["swift", "build", "-c", "release", "--package-path", str(FLUID_DIR), *local]])


def setup_sherpa(label: str, model: str):
    name, sha256 = SHERPA[model]["name"], SHERPA[model]["sha256"]
    folder = ENGINES_DIR / "sherpa-onnx"
    if (folder / name / ".done").exists():
        return

    archive = folder / f"{name}.tar.bz2"
    if not archive.exists():
        get_file(label, f"{SHERPA_URL}/{name}.tar.bz2", archive, sha256)

    with tarfile.open(archive) as tar:
        tar.extractall(folder, filter="data")

    (folder / name / ".done").touch()
    archive.unlink()


def setup_parakeet(label: str):
    if not PARAKEET_SERVER.exists():
        flags = ["-DPARAKEET_GGML_METAL=ON"]
        cmake_build(label, "parakeet.cpp", PARAKEET_URL, PARAKEET_TAG, "parakeet-server", flags)

    if not PARAKEET_MODEL.exists():
        get_file(label, PARAKEET_GGUF["url"], PARAKEET_MODEL, PARAKEET_GGUF["sha256"])


def setup(row: dict, label: str) -> str:
    """Whatever a row's engine still lacks; its version where this script pins it."""
    engine, model = row["engine"], row["model"]
    if engine.startswith("wavo"):
        version = build_wavo()
        gguf(model)
        return version

    if engine == "transcribe.cpp":
        build_wavo()
        if not TRANSCRIBE_CLI.exists():
            flags = ["-DTRANSCRIBE_BUILD_TESTS=OFF"]
            args = (TRANSCRIBE_URL, TRANSCRIBE_TAG, "transcribe-cli", flags)
            cmake_build(label, "transcribe.cpp", *args)

        gguf(model)
        return TRANSCRIBE_TAG

    if engine == "fluidaudio":
        setup_fluidaudio(label)
        return FLUID_TAG

    if engine == "parakeet.cpp":
        setup_parakeet(label)
        return PARAKEET_TAG

    if engine == "sherpa-onnx":
        setup_sherpa(label, model)

    return ""  # a Python engine says its own, from what uv installed


def command(row: dict, listed: Path, offline: bool) -> list[str]:
    """The process of a row over a list; `time` reports on it, wherever it is started."""
    engine, model = row["engine"], row["model"]
    if engine.startswith("wavo"):
        backend = engine.removeprefix("wavo-")
        return [TIME, "-l", str(BATCH), gguf(model), str(listed), backend]

    if engine == "transcribe.cpp":
        cmd = [str(TRANSCRIBE_CLI), "-m", gguf(model), "--backend", "metal", "-q"]
        return [TIME, "-l", *cmd, "--batch", str(listed), "--batch-jsonl"]

    if engine == "fluidaudio":
        return [TIME, "-l", str(FLUID_BIN), str(FLUID_DIR / "models"), str(listed)]

    worker = [str(SELF), "worker", engine, model, str(listed)]
    if engine == "parakeet.cpp":
        return [sys.executable, *worker]

    uv = ["uv", "run", *(["--offline"] if offline else []), "--with", PACKAGES[engine]]
    return [*uv, "--script", *worker]


def batch(row: dict, listed: Path, label: str, offline: bool = True) -> dict:
    """One process over a list: its load and call times, words, wall time and peak memory."""
    files = listed.read_text().splitlines()
    env = ENV | ({"HF_HUB_OFFLINE": "1"} if offline else {})
    log = WORK / "engine.log"
    head, calls, words = {}, [], 0
    with open(log, "w") as err, tqdm(total=len(files), desc=label, **FILES) as bar:
        cmd = command(row, listed, offline)
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err, env=env, encoding="utf-8", errors="replace"
        )
        for line in p.stdout:
            if not line.startswith("{"):
                continue

            x = json.loads(line)
            if x.get("type") == "batch_header":
                head = x
                continue

            parts = ("mel_ms", "encode_ms", "decode_ms")  # transcribe.cpp has no `ms`
            calls.append(round(x["ms"] if "ms" in x else sum(x[k] for k in parts), 1))
            words += len(x.get("text", "").split())
            bar.update()

        p.wait()

    stderr = log.read_text(errors="replace")
    real = re.search(r"([\d.]+) real", stderr)
    peak = re.search(r"(\d+)\s+peak memory footprint", stderr)
    if p.returncode or not peak or len(calls) != len(files):
        said = stderr[: real.start() if real else None].strip().splitlines()
        reason = f"exit code {p.returncode}, {len(calls)} of {len(files)} calls"
        raise RuntimeError(said[-1] if p.returncode and said else reason)

    return head | {
        "calls": calls,
        "words": words,
        "wall_s": float(real[1]),
        "peak_mb": round(int(peak[1]) / 1e6),
    }


# MARK: Workers


def performance_cores() -> int:
    cmd = ["sysctl", "-n", "hw.perflevel0.physicalcpu"]
    return int(subprocess.run(cmd, capture_output=True, text=True).stdout)


def load_mlx(model: str) -> tuple:
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel

    m = from_pretrained(MLX_MODEL)

    def call(file: str, pcm: np.ndarray) -> str:
        # `m.transcribe` without its ffmpeg step, which only gets `pcm` from the file.
        return m.generate(get_logmel(mx.array(pcm), m.preprocessor_config))[0].text

    return call, f"{package_version('parakeet-mlx')} mlx {package_version('mlx')}"


def load_sherpa(model: str) -> tuple:
    import sherpa_onnx

    folder = ENGINES_DIR / "sherpa-onnx" / SHERPA[model]["name"]
    parts = {x: str(next(folder.glob(f"{x}*.onnx"))) for x in ("encoder", "decoder", "joiner")}
    threads = performance_cores()
    r = sherpa_onnx.OfflineRecognizer.from_transducer(
        **parts,
        tokens=str(folder / "tokens.txt"),
        num_threads=threads,
        provider="cpu",
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )

    def call(file: str, pcm: np.ndarray) -> str:
        stream = r.create_stream()
        stream.accept_waveform(RATE, pcm)
        r.decode_stream(stream)
        return stream.result.text

    return call, f"{package_version('sherpa-onnx')} {threads} threads"


def load_parakeet(model: str) -> tuple:
    """parakeet-server under `time`, which reports on it instead of on this worker."""
    url = f"http://127.0.0.1:{PORT}"
    cmd = [TIME, "-l", str(PARAKEET_SERVER), "--model", str(PARAKEET_MODEL)]
    cmd += ["--host", "127.0.0.1", "--port", str(PORT)]
    # Its own group: SIGINT stops the server, `time` ignores it and stays to report.
    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, start_new_session=True)

    def stop():
        os.killpg(server.pid, signal.SIGINT)
        server.wait()

    atexit.register(stop)
    client = httpx.Client(timeout=600)
    while True:
        if server.poll() is not None:
            atexit.unregister(stop)
            sys.exit(f"parakeet-server stopped with code {server.returncode}")

        try:
            client.get(f"{url}/health").raise_for_status()
            break
        except httpx.TransportError:
            time.sleep(0.01)

    def call(file: str, pcm: np.ndarray) -> str:
        audio = Path(file).read_bytes()
        r = client.post(f"{url}/v1/audio/transcriptions", files={"file": ("audio.wav", audio)})
        r.raise_for_status()
        return r.json()["text"]

    return call, PARAKEET_TAG


def worker(args):
    """One engine over a list, printing what `transcribe-cli --batch --batch-jsonl` prints."""
    if args.engine in PACKAGES and not args.timed:  # `time` must watch this Python, not uv
        cmd = [sys.executable, str(SELF), "worker", "--timed", args.engine, args.model, args.list]
        os.execv(TIME, [TIME, "-l", *cmd])

    loaders = {"parakeet-mlx": load_mlx, "sherpa-onnx": load_sherpa, "parakeet.cpp": load_parakeet}
    start = time.perf_counter()
    call, version = loaders[args.engine](args.model)
    load_ms = (time.perf_counter() - start) * 1e3
    head = {"type": "batch_header", "load_ms": round(load_ms, 1), "version": version}
    print(json.dumps(head), flush=True)
    for file in Path(args.list).read_text().splitlines():
        pcm = sf.read(file, dtype="float32")[0]
        start = time.perf_counter()
        text = call(file, pcm)
        ms = (time.perf_counter() - start) * 1e3
        print(json.dumps({"file": file, "text": text, "ms": round(ms, 1)}), flush=True)


# MARK: Measurements


def attempt(row: dict, step) -> str:
    """A step's line; a failed one leaves its reason in the row, and the others go on."""
    try:
        return step()
    except Exception as e:
        row["error"] = str(e)
        return "failed"


def ready(row: dict, label: str) -> str:
    """A row's engine built and its model in place, out of the times, with one clip through it."""
    version = setup(row, label)
    r = batch(row, clip_list(row["language"], SECONDS[0]), label, offline=False)
    version = r.get("version", version)
    if row["version"] != version:
        row |= {"version": version, "latency": {}, "hour": {"runs": []}}

    return version


def latency(row: dict, seconds: int, label: str) -> str:
    r = batch(row, clip_list(row["language"], seconds), label)
    first, warm = r["calls"][0], median(r["calls"][1:])
    found = {"load_ms": r["load_ms"], "first_ms": first, "warm_ms": warm, "calls": r["calls"]}
    row["latency"][f"{seconds}s"] = found
    return f"load {r['load_ms']:>5.0f} first {first:>5.0f} warm {warm:>5.0f} ms"


def hour(row: dict, label: str) -> str:
    listed = hour_list(row["language"])
    files = listed.read_text().splitlines()
    audio_s = sum(sf.info(x).duration for x in files)
    for file in files:  # into the page cache, so no run reads disk
        Path(file).read_bytes()

    r = batch(row, listed, label)
    wall_s, peak_mb, words = r["wall_s"], r["peak_mb"], r["words"]
    run = {"wall_s": wall_s, "realtime": round(audio_s / wall_s, 2)}
    run |= {"peak_mb": peak_mb, "words": words}
    found = row["hour"]
    found["runs"].append(run)
    found |= {"audio_s": round(audio_s, 1), "pieces": len(files)}
    found |= {k: round(median(x[k] for x in found["runs"]), 2) for k in run}
    return f"{wall_s:>7.1f} s {run['realtime']:>6.1f}x {peak_mb:>5} MB {words:>6} w"


def cooldown(label: str, seconds: int):
    for _ in tqdm(range(seconds), desc=label, **STEPS):
        time.sleep(1)


def cell(x: float | None, width: int, digits: int = 0) -> str:
    return f"{'-':>{width}}" if x is None else f"{x:>{width}.{digits}f}"


def table(rows: list[dict]):
    for model, lang in LANGUAGE.items():
        mine = [x for x in rows if x["model"] == model]
        if not mine:
            continue

        print(f"\n{model}, {lang}: hour, median of {RUNS} runs")
        who = f"{'engine':<14} {'version':<24} {'backend':<13} {'precision':<12}"
        print(f"{who} {'wall s':>8} {'realtime':>8} {'peak MB':>7} {'words':>6} {'/wavo-gpu':>9}")
        base = next((x["hour"].get("wall_s") for x in mine if x["engine"] == "wavo-gpu"), None)
        for x in mine:
            h = x["hour"]
            wall = h.get("wall_s")
            who = f"{x['engine']:<14} {x['version'] or '-':<24} {x['backend']:<13}"
            print(
                f"{who} {x['precision']:<12} {cell(wall, 8, 1)} {cell(h.get('realtime'), 8, 1)}",
                f"{cell(h.get('peak_mb'), 7)} {cell(h.get('words'), 6)}",
                cell(wall / base if wall and base else None, 9, 2),
            )

        print(f"\n{model}, {lang}: latency in ms, {CALLS} calls of each clip in one process")
        clips = "".join(f"{f'{x}s':>7}" for x in SECONDS)
        print(f"{'':<14} {'':>7} {'first':<{7 * len(SECONDS)}} warm")
        print(f"{'engine':<14} {'load':>7} {clips} {clips}")
        for x in mine:
            found = [x["latency"].get(f"{s}s", {}) for s in SECONDS]
            load = [y["load_ms"] for y in found if y]
            first = "".join(cell(y.get("first_ms"), 7) for y in found)
            warm = "".join(cell(y.get("warm_ms"), 7) for y in found)
            print(f"{x['engine']:<14} {cell(median(load) if load else None, 7)} {first} {warm}")

    print()
    for engine in dict.fromkeys(x["engine"] for x in rows):
        print(f"{engine:<14} {ENGINES[engine]['notes']}")


def chip() -> str:
    cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def chosen_rows(args) -> list[dict]:
    """A row for each model and engine that runs it, wavo first."""
    engines = list(ENGINES) if args.engines == "all" else args.engines.split(",")
    models = list(LANGUAGE) if args.models == "all" else args.models.split(",")
    bad = [x for x in engines if x not in ENGINES] + [x for x in models if x not in LANGUAGE]
    if bad:
        sys.exit(f"unknown {', '.join(bad)}: see ENGINES and LANGUAGE in {SELF.name}")

    rows = []
    for model in models:
        for engine in engines:
            info = ENGINES[engine]
            if model not in info["precision"]:
                continue

            row = {"engine": engine, "model": model, "language": LANGUAGE[model], "version": None}
            row |= {"backend": info["backend"], "precision": info["precision"][model]}
            row |= {"notes": info["notes"], "latency": {}, "hour": {"runs": []}}
            rows.append(row)

    return rows


def key(row: dict) -> tuple[str, str]:
    return row["engine"], row["model"]


def run(args):
    if sys.platform != "darwin":
        sys.exit("macOS only: the engines compared here are Apple Silicon's")

    chosen = chosen_rows(args)
    if args.clean:
        print("--clean deletes the saved results of these rows and starts over:")
        print("".join(f"  {x['model']} {x['engine']}\n" for x in chosen), end="")
        if input("Type yes to go on: ") != "yes":
            sys.exit("--clean not confirmed: run without it to continue, or type yes")

    get_audio()
    name = re.sub(r"[^a-z0-9]+", "-", chip().lower()).strip("-")
    out = REPO / "results" / f"speed-{name}.json"
    out.parent.mkdir(exist_ok=True)
    old = json.loads(out.read_text())["rows"] if out.exists() else []
    rows = {key(x): x for x in old}
    for row in chosen:
        kept = {} if args.clean else rows.get(key(row), {})
        names = ("version", "latency", "hour")
        rows[key(row)] = row | {k: v for k, v in kept.items() if k in names}

    work = [rows[key(x)] for x in chosen]
    date = datetime.now(UTC).date().isoformat()
    report = {"chip": chip(), "macos": platform.mac_ver()[0], "date": date}

    def save():
        saved = report | {"rows": list(rows.values())}
        out.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")

    for n, row in enumerate(work, 1):
        label = f"{n:>3}/{len(work):>3} {row['model']:<11} {row['engine']:<14} {'ready':<6}"
        print(f"{label}  {attempt(row, lambda: ready(row, label))}")
        save()

    good = [x for x in work if "error" not in x]
    plan = [(x, f"{s}s", s) for x in good for s in SECONDS if f"{s}s" not in x["latency"]]
    rounds = range(1, RUNS + 1)
    plan += [(x, f"hour {r}", None) for r in rounds for x in good if len(x["hour"]["runs"]) < r]
    rest = False
    for n, (row, name, seconds) in enumerate(plan, 1):
        if "error" in row:
            continue

        label = f"{n:>3}/{len(plan):>3} {row['model']:<11} {row['engine']:<14} {name:<6}"
        if seconds:
            print(f"{label}  {attempt(row, lambda: latency(row, seconds, label))}")
        else:
            if rest:
                cooldown(label, args.cooldown)

            rest = True
            print(f"{label}  {attempt(row, lambda: hour(row, label))}")

        save()

    table(list(rows.values()))
    print(out)
    failed = [x for x in work if "error" in x]
    for x in failed:
        print(f"failed: {x['model']} {x['engine']}: {x['error']}")

    if failed:
        sys.exit(f"{len(failed)} rows failed: run it again to retry them")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{download,run}")
    x = sub.add_parser("download")
    x.set_defaults(fn=download)

    x = sub.add_parser("run")
    x.add_argument("--engines", default="all", help=f"{','.join(ENGINES)}, or all")
    x.add_argument("--models", default="all", help=f"{','.join(LANGUAGE)}, or all")
    x.add_argument("--cooldown", type=int, default=30, help="seconds of rest between hour runs")
    x.add_argument("--clean", action="store_true", help="start these rows over: old results go")
    x.set_defaults(fn=run)

    x = sub.add_parser("worker")  # what `run` starts for an engine with no batch mode of its own
    x.add_argument("--timed", action="store_true")
    x.add_argument("engine")
    x.add_argument("model")
    x.add_argument("list")
    x.set_defaults(fn=worker)

    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit("\nstopped: run it again to continue")
    except Exception as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
