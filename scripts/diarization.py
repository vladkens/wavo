#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.28.1", "pyannote-metrics>=4.2.0", "tqdm>=4.70.1",
# ]
# ///
"""Who speaks when: wavo against other diarization engines on this Mac, on public meetings.

    scripts/diarization.py run [--engines wavo,wavo-cpu,fluidaudio,fluidaudio-pyannote,pyannote|all]
                               [--clean]

`run` fetches what is missing into `.cache/data/`: AMI's 16 test meetings (Mix-Headset, 9.1 h, 3-4
speakers, 1.1 GB) and two sets of labels. NVIDIA's model card scores against forced-alignment ones
(nttcslab-sp/diar-forced-alignment: word times from an alignment, the pauses between words left
out), pyannote against manual ones (pyannote/AMI-diarization-setup, only_words: the annotators'
word times); the ranking turns on which. Not VoxConverse: NVIDIA trained the model on its dev and
test sets. It builds each engine into `.cache/engines/`, runs it over the meetings in one process,
keeps the turns in `.cache/diarization/` and continues where it stopped, then writes
`results/diarization-<machine>.json` and prints the table.

The error is pyannote.metrics' DER at collar 0 with overlapping speech counted: the share of the
reference's speech time with a missed speaker, a false one or the wrong one, all files together.
Speed is seconds of audio per second of the calls: wavo's include the model load and the audio
read of each file, the others' don't. pyannote needs a Hugging Face token (`hf auth login`) of an
account that accepted the terms of pyannote/speaker-diarization-community-1.
"""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
DATA = REPO / ".cache" / "data"
WORK = REPO / ".cache" / "diarization"
ENGINES_DIR = REPO / ".cache" / "engines"
WAVO = REPO / "target" / "release" / "wavo"
BAR = {"ncols": 80, "leave": False}
FILES = BAR | {"unit": "file", "bar_format": "{desc} |{bar}| {n:>4} / {total:>4} {rate_fmt:>10}"}

AMI_AUDIO = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{0}/audio/{0}.Mix-Headset.wav"
)
AMI_LABELS = (
    "https://raw.githubusercontent.com/nttcslab-sp/diar-forced-alignment/"
    "9527b7c64846fb38316a610f32e9d3466bd6d8b7/AMI/test/{}.rttm"
)
AMI_MANUAL = (
    "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/"
    "67c2d539286e89f68952d5dcf83912bd9f01dfae/only_words/rttms/test/{}.rttm"
)
AMI = {
    "EN2002a": "a15017d0d3c871866971ee178052a0ce3a7e44d6fbf89fc23dcab326511b65f5",
    "EN2002b": "50e2c8e4737850c4bf3cc24e7087c755e1c733cd91822106afd0df485ba64893",
    "EN2002c": "a057708b3adb553f709fcde97d2af8e0455076bc3f88c250975227433d6e0270",
    "EN2002d": "6aa99d9d71cb695ea699d6be277f78eff68e847c93560031486eba9f18164ebe",
    "ES2004a": "3e2560b19bee6952c7c7ce041b0f1ea8a7ea9468044c4eea79d2a2c67e24ab0f",
    "ES2004b": "ad0cf07c42b1694ccf7bea8f1a37348d9cfab194787abc7d050b29ba56e365c8",
    "ES2004c": "47dc9bd2e3b1d174842091ae9e1fe765218f0c87658b17d6469c404e03ec2ac5",
    "ES2004d": "3cfce3297ebb7faeb53c02727eea5e9f2c76e9ed4b62db3b03d73e66836e91f3",
    "IS1009a": "6eb5a0ede0d9e72794f976ce7bea5b78133eae969f99b4c5418b43c2468d25b1",
    "IS1009b": "07c2891ae6ad7c507b2a4f15b2dcd0a2343491df9e4e6a58f9db0d0b2c5e1382",
    "IS1009c": "f317910070db0f2649315decbb34686d648bbe9dc3f7576710184f4d572bdbcc",
    "IS1009d": "470440114ee2ed076cfc77c53136ad5d3af500d8a5a6ca36755e83534f5ca4ea",
    "TS3003a": "b6ce59ff735c234afdd94b13744aa26f6785af746170543fd03b33160e920a6d",
    "TS3003b": "0c94f3a09ab747caa7714efe8852f5ff37d36cf272b75709344991df1aa266ca",
    "TS3003c": "b9f455870e8cb0ba765523a1767cc366775ab7e5e3eabf37d9c91289e5a1a70a",
    "TS3003d": "ece4a41c37c1092c737d1dfe313afb18248364625991f745fed4b372c3c67eb3",
}
# What the turns are scored against: a name for the results, the labels' file suffix, a title.
LABELS = {
    "ami": (".json", "AMI test, forced-alignment labels"),
    "ami-manual": (".manual.json", "AMI test, manual labels"),
}

FLUID_TAG = "v0.17.7"
FLUID_DIR = ENGINES_DIR / "fluid-diar"
FLUID_BIN = FLUID_DIR / ".build" / "release" / "fluid-diar"
FLUID_PACKAGE = f"""// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "fluid-diar",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio", exact: "{FLUID_TAG[1:]}")
    ],
    targets: [
        .executableTarget(
            name: "fluid-diar",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]
        )
    ]
)
"""
# `fluid-diar nemotron|pyannote LIST`: a JSON line per file with its turns and the call's
# milliseconds, the audio read before the clock starts. Nemotron with the model card's offline
# config, the one wavo has; pyannote is FluidAudio's CoreML port of community-1 with its defaults.
FLUID_MAIN = """import FluidAudio
import Foundation

func emit(_ fields: [String: Any]) throws {
    var line = try JSONSerialization.data(withJSONObject: fields)
    line.append(10)
    FileHandle.standardOutput.write(line)
}

func ms(since start: Date) -> Double { Date().timeIntervalSince(start) * 1000 }

@main
struct Main {
    static func main() async throws {
        let args = CommandLine.arguments
        let files = try String(contentsOfFile: args[2], encoding: .utf8).split(separator: "\\n")
        let converter = AudioConverter()
        if args[1] == "nemotron" {
            let config = Nemotron3Config.offline
            let models = try await Nemotron3Models.loadFromHuggingFace(config: config)
            let diarizer = Nemotron3Diarizer(config: config, models: models)
            for file in files {
                let audio = try converter.resampleAudioFile(path: String(file))
                let start = Date()
                let (probs, frames) = try diarizer.processComplete(audio)
                let segs = Nemotron3Diarizer.segments(probabilities: probs, frameCount: frames)
                let t = ms(since: start)
                try emit([
                    "file": String(file), "ms": t,
                    "turns": segs.map { [String($0.speakerIndex), $0.startSeconds, $0.endSeconds] },
                ])
            }
        } else {
            let manager = OfflineDiarizerManager(config: OfflineDiarizerConfig())
            manager.initialize(models: try await OfflineDiarizerModels.load())
            for file in files {
                let audio = try converter.resampleAudioFile(path: String(file))
                let start = Date()
                let result = try await manager.process(audio: audio)
                let t = ms(since: start)
                try emit([
                    "file": String(file), "ms": t,
                    "turns": result.segments.map { [$0.speakerId, $0.startTimeSeconds, $0.endTimeSeconds] },
                ])
            }
        }
    }
}
"""
PYANNOTE = "pyannote.audio==4.0.7"
PYANNOTE_DIR = ENGINES_DIR / "pyannote"
# `run.py LIST`: the same lines through PyTorch on the GPU (MPS), as WhisperX runs it.
PYANNOTE_MAIN = """import json, sys, time
import soundfile as sf
import torch
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
pipe.to(torch.device("mps"))
for file in open(sys.argv[1]).read().split():
    pcm, rate = sf.read(file, dtype="float32")
    audio = {"waveform": torch.from_numpy(pcm)[None], "sample_rate": rate}
    start = time.monotonic()
    out = pipe(audio).speaker_diarization
    ms = 1000 * (time.monotonic() - start)
    turns = [[s, t.start, t.end] for t, _, s in out.itertracks(yield_label=True)]
    print(json.dumps({"file": file, "ms": ms, "turns": turns}), flush=True)
"""

ENGINES = {
    "wavo": {"on": "GPU", "model": "Nemotron 3 Diarization, Q8_0"},
    "wavo-cpu": {"on": "CPU", "model": "Nemotron 3 Diarization, Q8_0"},
    "fluidaudio": {"on": "ANE", "model": "Nemotron 3 Diarization, CoreML"},
    "fluidaudio-pyannote": {"on": "ANE", "model": "pyannote community-1, CoreML"},
    "pyannote": {"on": "GPU", "model": "pyannote community-1, PyTorch"},
}


# MARK: Data


def get_file(url: str, to: Path, sha256: str | None = None):
    """`url` into `to`, which takes its name once its hash, if given, is right."""
    to.parent.mkdir(parents=True, exist_ok=True)
    part = to.with_suffix(".part")
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        with client.stream("GET", url) as r, open(part, "wb") as f:
            r.raise_for_status()
            f.writelines(r.iter_bytes(1 << 20))

    if sha256 and (digest := hashlib.sha256(part.read_bytes()).hexdigest()) != sha256:
        part.unlink()
        raise RuntimeError(f"{to.name}: sha256 {digest}, expected {sha256}")

    part.rename(to)


def get_ami() -> Path:
    folder = DATA / "ami-test"
    for name, sha256 in tqdm(AMI.items(), desc="AMI test", **FILES):
        if not (folder / f"{name}.wav").exists():
            get_file(AMI_AUDIO.format(name), folder / f"{name}.wav", sha256)

        for url, suffix in [(AMI_LABELS, ".json"), (AMI_MANUAL, ".manual.json")]:
            if not (folder / f"{name}{suffix}").exists():
                get_file(url.format(name), folder / f"{name}.rttm")
                rows = [x.split() for x in (folder / f"{name}.rttm").read_text().splitlines()]
                turns = [[r[7], float(r[3]), float(r[3]) + float(r[4])] for r in rows if r]
                (folder / f"{name}{suffix}").write_text(json.dumps(turns))
                (folder / f"{name}.rttm").unlink()

    return folder


# MARK: Engines


def sh(cmd: list[str], **kw) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        said = (r.stderr or r.stdout).strip().splitlines()
        raise RuntimeError(f"{Path(cmd[0]).name}: {said[-1] if said else r.returncode}")

    return r.stdout


def setup(engine: str) -> str:
    """The engine built, its version."""
    if engine.startswith("wavo"):
        sh(["cargo", "build", "--release", "--locked", "--bin", "wavo"], cwd=REPO)
        sh([str(WAVO), "pull", "nemotron-diarization"])
        rev = sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).strip()
        return f"{sh([str(WAVO), '--version']).split()[-1]} {rev}"

    if engine.startswith("fluidaudio"):
        if not FLUID_BIN.exists():
            (FLUID_DIR / "Sources" / "fluid-diar").mkdir(parents=True, exist_ok=True)
            (FLUID_DIR / "Package.swift").write_text(FLUID_PACKAGE)
            (FLUID_DIR / "Sources" / "fluid-diar" / "Main.swift").write_text(FLUID_MAIN)
            local = [
                f"--{x}-path={FLUID_DIR / 'swiftpm' / x}" for x in ("cache", "config", "security")
            ]
            sh(["swift", "build", "-c", "release", "--package-path", str(FLUID_DIR), *local])

        return FLUID_TAG[1:]

    PYANNOTE_DIR.mkdir(parents=True, exist_ok=True)
    (PYANNOTE_DIR / "run.py").write_text(PYANNOTE_MAIN)
    return PYANNOTE.split("==")[1]


def command(engine: str, listed: Path) -> list[str]:
    if engine == "fluidaudio":
        return [str(FLUID_BIN), "nemotron", str(listed)]

    if engine == "fluidaudio-pyannote":
        return [str(FLUID_BIN), "pyannote", str(listed)]

    uv = ["uv", "run", "--no-project", "-p", "3.12", "--with", PYANNOTE, "--with", "soundfile"]
    return [*uv, "python", "-I", str(PYANNOTE_DIR / "run.py"), str(listed)]


def wavo_turns(engine: str, wav: Path) -> dict:
    """wavo's CLI on one file, timed from start to exit."""
    on = ["--backend", "cpu"] if engine == "wavo-cpu" else []
    start = time.monotonic()
    out = sh([str(WAVO), "run", "nemotron-diarization", str(wav), "--json", *on])
    ms = 1000 * (time.monotonic() - start)
    turns = [
        [t["speaker"], t["start_ms"] / 1000, t["end_ms"] / 1000] for t in json.loads(out)["turns"]
    ]
    return {"ms": ms, "turns": turns}


def diarize(engine: str, wavs: list[Path], saved: Path, label: str):
    """The turns of the files not in `saved` yet, a JSON line each as it comes."""
    with open(saved, "a") as out, tqdm(total=len(wavs), desc=label, **FILES) as bar:
        if engine.startswith("wavo"):
            for wav in wavs:
                out.write(json.dumps({"name": wav.stem} | wavo_turns(engine, wav)) + "\n")
                out.flush()
                bar.update()

            return

        listed = WORK / "list.txt"
        listed.write_text("".join(f"{x}\n" for x in wavs))
        log = WORK / f"{engine}.log"
        with open(log, "w") as err:
            cmd = command(engine, listed)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
            for line in p.stdout:
                if line.startswith("{"):
                    x = json.loads(line)
                    out.write(json.dumps({"name": Path(x.pop("file")).stem} | x) + "\n")
                    out.flush()
                    bar.update()

        if p.wait():
            said = log.read_text().strip().splitlines()
            raise RuntimeError(said[-1] if said else f"exit {p.returncode}")


# MARK: Scores


def annotation(turns: list) -> Annotation:
    a = Annotation()
    for n, (speaker, start, end) in enumerate(turns):
        if end > start:
            a[Segment(start, end), n] = str(speaker)

    return a


def seconds(wav: Path) -> float:
    with wave.open(str(wav)) as f:
        return f.getnframes() / f.getframerate()


def score(folder: Path, saved: Path, labels: str) -> dict:
    """The turns in `saved` against the labels of their files, those with the suffix `labels`."""
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    parts = dict.fromkeys(["missed detection", "false alarm", "confusion", "total"], 0.0)
    right, audio_s, time_s = 0, 0.0, 0.0
    runs = [json.loads(x) for x in saved.read_text().splitlines()]
    for x in runs:
        ref = annotation(json.loads((folder / f"{x['name']}{labels}").read_text()))
        length = seconds(folder / f"{x['name']}.wav")
        got = metric(ref, annotation(x["turns"]), uem=Timeline([Segment(0, length)]), detailed=True)
        parts = {k: v + got[k] for k, v in parts.items()}
        right += len(ref.labels()) == len({str(t[0]) for t in x["turns"]})
        audio_s += length
        time_s += x["ms"] / 1000

    share = lambda k: round(100 * parts[k] / parts["total"], 1)
    return {
        "files": len(runs),
        "audio_s": round(audio_s, 1),
        "time_s": round(time_s, 1),
        "speed": round(audio_s / time_s),
        "der": round(100 * abs(metric), 1),
        "missed": share("missed detection"),
        "false_alarm": share("false alarm"),
        "confusion": share("confusion"),
        "speakers_right": right,
    }


def table(rows: list[dict]):
    for name in dict.fromkeys(x["set"] for x in rows):
        print(f"\n{LABELS[name][1]}  DER, collar 0: missed, false alarm, wrong speaker")
        print(
            f"{'engine':<20} {'on':<4} {'DER':>6} {'miss':>5} {'FA':>5} {'conf':>5} {'speakers':>9} {'speed':>7}"
        )
        for x in sorted(
            (x for x in rows if x["set"] == name and "der" in x), key=lambda x: x["der"]
        ):
            right = f"{x['speakers_right']}/{x['files']}"
            print(
                f"{x['engine']:<20} {x['on']:<4} {x['der']:>5}% {x['missed']:>5} {x['false_alarm']:>5}"
                f" {x['confusion']:>5} {right:>9} {x['speed']:>6}×"
            )


def machine() -> str:
    """What a results file is named after, as in scripts/speed.py: `mac-pro-m5`."""
    info = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True)
    model = re.search(r"Model Name: (.+)", info.stdout)[1]
    chip = re.search(r"Chip: (.+)", info.stdout)[1]
    name = f"{model} {chip}".replace("MacBook", "Mac").replace("Apple ", "")
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def run(args):
    if sys.platform != "darwin":
        sys.exit("macOS only: FluidAudio runs on Apple Silicon")

    engines = list(ENGINES) if args.engines == "all" else args.engines.split(",")
    bad = [x for x in engines if x not in ENGINES]
    if bad:
        sys.exit(f"unknown {', '.join(bad)}: see ENGINES in {SELF.name}")

    folder = get_ami()
    out = REPO / "results" / f"diarization-{machine()}.json"
    old = json.loads(out.read_text())["rows"] if out.exists() else []
    rows = {(x["set"], x["engine"]): x for x in old}
    chip = sh(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    date = datetime.now(UTC).date().isoformat()
    report = {"machine": machine(), "chip": chip, "macos": platform.mac_ver()[0], "date": date}
    failed = []
    for engine in engines:
        try:
            version = setup(engine)
        except Exception as e:
            failed.append(f"{engine}: {e}")
            continue

        saved = WORK / f"{engine}.jsonl"
        # Turns of another version of the engine go.
        stamp = saved.with_suffix(".version")
        if args.clean or (stamp.exists() and stamp.read_text() != version):
            saved.unlink(missing_ok=True)

        stamp.write_text(version)

        done = (
            {json.loads(x)["name"] for x in saved.read_text().splitlines()}
            if saved.exists()
            else set()
        )
        wavs = [x for x in sorted(folder.glob("*.wav")) if x.stem not in done]
        try:
            if wavs:
                diarize(engine, wavs, saved, engine)
        except Exception as e:
            failed.append(f"{engine}: {e}")
            continue

        for name, (labels, _) in LABELS.items():
            rows[(name, engine)] = (
                {"set": name, "engine": engine, "version": version}
                | ENGINES[engine]
                | score(folder, saved, labels)
            )

        saved_rows = sorted(rows.values(), key=lambda x: (x["set"], x["der"]))
        out.write_text(
            json.dumps(report | {"rows": saved_rows}, ensure_ascii=False, indent=2) + "\n"
        )

    table(list(rows.values()))
    print(f"\n{out}")
    if failed:
        sys.exit("failed:\n" + "\n".join(failed) + "\nrun it again to retry them")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{run}")
    x = sub.add_parser("run")
    x.add_argument("--engines", default="all", help=f"{','.join(ENGINES)}, or all")
    x.add_argument("--clean", action="store_true", help="run these engines over again")
    x.set_defaults(fn=run)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        sys.exit("\nstopped: run it again to continue")


if __name__ == "__main__":
    main()
