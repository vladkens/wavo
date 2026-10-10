# Performance notes

Apple M2 (8 CPU and 10 GPU cores, 24 GiB), macOS 27, wgpu 30.0.1 Metal. Times are warm medians
without model load unless stated. GigaAM clips: `ru` 4.5 s, `ru-short` 11 s, `ru-long` 33.8 s.
Parakeet clips: `jfk` 11 s, `dots` 35.3 s, `jobs-silence` 5.3 s; for V3 `jfk`, `ru-short` and
`uk-short`, 11 s each.

## Benchmark: all models vs transcribe.cpp (2026-10-10, end of phase 6)

On the M2 above. Each cell is wavo / reference, ms (MiB for the footprint), on the same Q8_0 GGUF;
transcribe.cpp at commit 5bb2deb on Metal (`transcribe-bench`). Per model and clip, two rounds of:
`wavo bench -n 20` (warm median and min of 20 calls; first call and load from the same fresh
process), `transcribe-bench --warmup 1 --iters 10` (median and min of `wall_ms`), `/usr/bin/time -l
wavo MODEL AUDIO` (peak footprint), `transcribe-bench --warmup 0 --iters 1` under `/usr/bin/time -l`
(first call, load, peak footprint). wavo and the reference alternate step by step, so background
load hits both alike; the page cache and both engines' shader caches were warm. Ranges are the two
rounds. On AC power, load average 1.7–3.9 (a VM at ~15% of a core, no other work); 3 minutes in all.
That session ended phase 5; the GigaAM rows are from it. The Parakeet rows are a second session at
the end of phase 6 (Q8_0 decoder weights), same protocol and reference build: load average 2.6–4.0
with the VM at up to ~120% of a core; 2 minutes.

| Model | Clip | Warm median | Warm min | First call | Load | Peak footprint |
|---|---|---|---|---|---|---|
| GigaAM e2e-rnnt | ru | 43.5–44.1 / 43.7–45.4 | 43.3 / 43.7 | 46.6–47.0 / 46.5–48.7 | 76–78 / 122–124 | 294 / 323 |
| GigaAM e2e-rnnt | ru-short | 95.8–96.4 / 99.2–101.0 | 95.4 / 98.4 | 99.2–101.4 / 101.1–103.4 | 76–78 / 122–125 | 302 / 325–326 |
| GigaAM e2e-rnnt | ru-long | 299.3 / 310.4 | 296.4 / 308.7 | 301.4–302.9 / 310.5–334.7 | 76–77 / 122–128 | 321–322 / 334 |
| GigaAM e2e-ctc | ru | 39.9–40.3 / 41.8 | 39.7 / 41.3 | 43.2–43.5 / 46.9–59.4 | 74–78 / 123–136 | 289–290 / 313–314 |
| GigaAM e2e-ctc | ru-short | 85.5 / 92.6–92.7 | 85.1 / 91.9 | 88.7–89.0 / 95.0–96.3 | 75 / 121–130 | 295 / 315 |
| GigaAM e2e-ctc | ru-long | 270.0–270.3 / 295.5–296.1 | 269.0 / 292.8 | 275.3–276.2 / 303.9–312.0 | 74–77 / 122–131 | 314–316 / 323–324 |
| GigaAM rnnt | ru | 42.6 / 42.7–43.3 | 42.3 / 42.6 | 45.7–46.1 / 46.3–54.6 | 74–75 / 123–127 | 291 / 318–319 |
| GigaAM rnnt | ru-short | 96.1 / 97.1–97.7 | 95.5 / 96.2 | 99.0–99.4 / 99.0–100.1 | 76 / 121–123 | 298–299 / 321–322 |
| GigaAM rnnt | ru-long | 296.2–296.5 / 305.7–306.3 | 295.1 / 304.2 | 299.8–301.0 / 308.8–322.1 | 76 / 123–126 | 320 / 329–330 |
| GigaAM ctc | ru | 39.7–39.8 / 41.3–41.4 | 39.5 / 41.1 | 43.0–43.3 / 44.3–60.3 | 73–75 / 121–126 | 287–289 / 312–314 |
| GigaAM ctc | ru-short | 85.0 / 91.5–91.6 | 84.9 / 90.5 | 89.0–89.9 / 93.0–95.1 | 74–75 / 125–132 | 293 / 314 |
| GigaAM ctc | ru-long | 269.0–269.2 / 290.2–292.5 | 268.2 / 288.8 | 273.5–275.8 / 289.5–301.1 | 75–79 / 123–129 | 314 / 322 |
| Parakeet TDT V2 | jfk | 142.2–142.9 / 198.1–200.3 | 141.2 / 191.1 | 147.1–150.0 / 204.7–259.0 | 158–159 / 272–276 | 740 / 820 |
| Parakeet TDT V2 | dots | 459.7–461.5 / 691.8–706.9 | 453.3 / 664.0 | 467.8–483.6 / 675.3–857.7 | 158–175 / 270–280 | 766–768 / 833–834 |
| Parakeet TDT V2 | jobs-silence | 83.4 / 95.4–98.9 | 82.9 / 94.0 | 87.3–88.0 / 115.7–115.8 | 168–181 / 274–285 | 734–736 / 816 |
| Parakeet TDT V3 | jfk | 153.0–155.2 / 212.7–218.2 | 148.5 / 209.2 | 155.9–161.5 / 277.9–284.2 | 167–170 / 280–287 | 762–764 / 883–884 |
| Parakeet TDT V3 | ru-short | 159.7–162.3 / 238.5–253.0 | 158.5 / 235.8 | 172.9 / 321.6–376.6 | 165–170 / 280–290 | 762–763 / 884 |
| Parakeet TDT V3 | uk-short | 159.1–175.6 / 232.6–275.3 | 157.1 / 228.9 | 169.4–170.6 / 273.6–333.8 | 184–192 / 281–354 | 761–762 / 884 |

- Warm median is faster on all 18: GigaAM by 1–9% (e2e-rnnt 2–4%, the CTC models 4–9%, rnnt
  1–3%), Parakeet V2 by 14–34%, V3 by 28–35%. Warm minimum too. In the second uk-short round both
  engines hit a background burst (wavo 175.6 against a minimum of 159.2 ms, the reference 275.3
  against 233.2); a second full Parakeet run right after (load average 3.3–5.9) gave V3 149.3–150.6
  / 159.7–160.6 / 158.6–159.0 ms against 241.6–243.8 / 249.0–279.0 / 237.7–259.8 ms.
- First call is on par with or below the reference everywhere; the closest are e2e-rnnt ru
  (46.6–47.0 vs 46.5–48.7) and rnnt ru-short (99.0–99.4 vs 99.0–100.1). Load is ~40% below for
  GigaAM (73–79 vs 121–136 ms) and 35–45% below for Parakeet (158–192 vs 270–354 ms); peak
  footprint is 8–29 MiB below for GigaAM, 66–82 MiB below for Parakeet V2 and 120–123 MiB below
  for V3.
- The reference ran faster than in earlier sessions (Parakeet dots 671–690 ms vs 799–821 on a
  machine at load average 3–5): its decoder runs on CPU threads and gains most from a quiet
  machine. wavo's numbers moved less (dots 492–504 ms here, 499–531 ms in the busier phase-5 A/B
  runs).

### Second machine: Apple M1 (8 CPU and 8 GPU cores, 16 GiB), macOS 15.8.1

Same protocol and binaries: wavo as built for the table above (minos 11.0), `transcribe-bench` at
5bb2deb built on the M2 with `-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0`; nothing compiled on the M1. All
15 transcripts' text equals the fixtures there. AC power, load average 1.3–3.0, no other work; 5
minutes in all (2026-10-10). This M1's fan is broken, so it throttles under sustained load: a
sanity check, not a reference for speed.

| Model | Clip | Warm median | Warm min | First call | Load | Peak footprint |
|---|---|---|---|---|---|---|
| GigaAM e2e-rnnt | ru | 75.2–76.5 / 74.9–75.4 | 58.9 / 61.1 | 60.1–60.8 / 67.1–67.5 | 83–91 / 134–148 | 294–296 / 332–333 |
| GigaAM e2e-rnnt | ru-short | 168.5–169.7 / 175.3–176.1 | 158.5 / 157.2 | 144.2–145.3 / 149.0–150.1 | 83–84 / 134–141 | 298–303 / 335–336 |
| GigaAM e2e-rnnt | ru-long | 452.3–452.8 / 461.8–463.7 | 449.3 / 457.1 | 450.1–450.5 / 461.6–464.6 | 85–86 / 133–138 | 322–324 / 341 |
| GigaAM e2e-ctc | ru | 57.4–57.5 / 72.0–73.9 | 51.4 / 56.3 | 54.9–55.1 / 63.4–64.4 | 82–102 / 132–155 | 285–286 / 322 |
| GigaAM e2e-ctc | ru-short | 129.4–129.5 / 148.3–148.7 | 120.5 / 138.2 | 122.2 / 134.3–135.4 | 82 / 135–157 | 291–293 / 323–324 |
| GigaAM e2e-ctc | ru-long | 396.6–397.2 / 436.6–438.4 | 392.9 / 432.9 | 379.7–381.0 / 421.9 | 86 / 145–157 | 315–316 / 328–330 |
| GigaAM rnnt | ru | 71.3–71.4 / 73.8–75.3 | 57.0 / 58.7 | 58.8–59.3 / 64.4–65.1 | 84–96 / 136–152 | 290 / 326–327 |
| GigaAM rnnt | ru-short | 166.9–169.4 / 167.0–168.0 | 153.9 / 151.2 | 143.6–143.9 / 142.8–143.4 | 83 / 134–141 | 297–299 / 329–330 |
| GigaAM rnnt | ru-long | 453.3–453.4 / 456.9–457.7 | 448.8 / 452.7 | 448.8–456.3 / 450.0–453.0 | 88 / 134–138 | 318–319 / 334–335 |
| GigaAM ctc | ru | 57.0–57.4 / 67.7–68.1 | 51.2 / 56.1 | 55.1–55.6 / 63.4 | 82–104 / 133–158 | 284–286 / 321 |
| GigaAM ctc | ru-short | 128.6 / 139.8–140.4 | 120.0 / 131.3 | 121.5–121.6 / 133.2–133.4 | 82 / 142–158 | 291–292 / 322 |
| GigaAM ctc | ru-long | 392.1–393.7 / 415.6–417.7 | 391.0 / 410.7 | 377.1–377.7 / 402.4–404.7 | 95 / 151–154 | 309–310 / 327–328 |
| Parakeet TDT V2 | jfk | 254.5 / 269.6–273.9 | 252.1 / 267.5 | 252.4–252.6 / 264.3–281.2 | 173–177 / 284–289 | 755–757 / 792–794 |
| Parakeet TDT V2 | dots | 719.5 / 882.9–883.0 | 716.1 / 870.9 | 725.1–725.4 / 883.3–895.5 | 176–177 / 283–286 | 777–779 / 810–812 |
| Parakeet TDT V2 | jobs-silence | 129.7–130.6 / 158.9–160.1 | 120.1 / 153.5 | 119.4–120.1 / 134.7–135.5 | 170 / 283–285 | 747–749 / 787–789 |

- Warm median: the CTC models 6–21% faster, Parakeet 6–19%, the two RNN-T GigaAM models on par
  (−1% to +4%: e2e-rnnt ru 75.2–76.5 vs 74.9–75.4, rnnt ru-short 166.9–169.4 vs 167.0–168.0).
  Warm minimum lower except e2e-rnnt and rnnt on ru-short (by 1.3 and 2.7 ms).
- Warm calls get slower than the first call for both engines on the M1, from the CPU side: on
  e2e-rnnt ru the reference's `encode_ms` stays at 50.5 per call while its `decode_ms` grows from
  4.9 to ~16 over 20 calls, and wavo's GPU wait stays at 50–51 ms while its mel + decoder grows
  from ~7 to ~24 ms. So on the M1 the RNN-T warm medians mostly measure each CPU decoder
  throttled. First call: wavo 3–10% faster on e2e-rnnt; on rnnt 9% faster on ru, on par on
  the longer clips.
- First call is below the reference everywhere except rnnt ru-short and ru-long (within 0.5%).
  Load is 35–45% below, peak footprint 14–40 MiB below.

### Linux: Intel N100 (4 cores, 24-EU UHD iGPU, 16 GiB), Ubuntu 24.04, Mesa 25.2.8

A fanless MeLE Quieter 4C. wavo picks the iGPU through Vulkan (ANV on i915): no cooperative
matrices, subgroups 8–32, so every kernel is the portable one. `default` is `cargo build
--release`, `native` adds `-C target-cpu=native`. transcribe.cpp 5bb2deb is its CPU build with
defaults (`-march=native`, 4 threads, no BLAS: "decoder uses scalar fallback"); its Vulkan build
needs `spirv-headers` and was not measured. Per clip, back to back: wavo default `bench -n 10`,
`transcribe-bench --warmup 1 --iters 10`, wavo native `bench -n 10`, `transcribe-bench --warmup 0
--iters 1` (first call, load, peak). Cells are wavo default / wavo native / reference, ms; peak
is RSS plus the iGPU buffers (DRM fdinfo) for wavo, RSS for the reference. Load average 1.0–1.7
(2026-10-10).

| Model | Clip | Warm median | First call | Load | Peak, MiB |
|---|---|---|---|---|---|
| GigaAM e2e-rnnt | ru | 963 / 828 / 902 | 964 / 833 / 898 | 1168 (cold shader cache) / 583 / 180 | 653 / 640 / 282 |
| GigaAM e2e-rnnt | ru-short | 2536 / 2133 / 2470 | 2568 / 2136 / 2484 | 584 / 614 / 182 | 640 / 641 / 289 |
| GigaAM e2e-rnnt | ru-long | 8456 / 7417 / 9847 (min 8478) | 8450 / 7433 / 8500 | 585 / 619 / 178 | 644 / 644 / 345 |
| Parakeet TDT V2 | jfk | 3857 / 3645 / 2537 | 3868 / 3653 / 2505 | 1380 / 1554 / 671 | 1038 / 1038 / 1098 |
| Parakeet TDT V2 | dots | 13628 / 12337 / 11045 (min 8929) | 13646 / 12337 / 8930 | 1394 / 1549 / 690 | 1046 / 1046 / 1212 |
| Parakeet TDT V2 | jobs-silence | 2183 / 2166 / 1225 | 2178 / 2167 / 1204 | 1446 / 1418 / 683 | 1036 / 1036 / 1073 |

- wavo matches the fixtures exactly (text, pieces, start times) on ANV and on llvmpipe, in both
  builds. The reference's CPU build matches the text on all six clips but moves two tokens each
  on ru-long and jfk by 1–3 frames.
- Per call, wavo native mel / encoder / decoder against the reference's mel / encode / decode
  medians: ru 0.7 / 815 / 11 vs 10.5 / 687 / 204, ru-long 5.5 / 7342 / 68 vs 78 / 8077 / 1690,
  jfk 5.5 / 3500 / 132 vs 41 / 2423 / 53, dots 25 / 11601 / 712 vs 168 / 10465 / 388,
  jobs-silence 6 / 2140 / 20 vs 21 / 1195 / 5. The portable encoder is 85–99% of wavo's time.
  The reference's CPU encoder quantizes activations to Q8_0 and multiplies in int8 (AVX-VNNI),
  and is faster; wavo wins GigaAM only through the reference's scalar RNN-T decoder.
- The default build's decoder is the plain-Rust fallback (`mul_add` without FMA is a libm call):
  GigaAM 145 / 419 / 1056 ms on the three clips, Parakeet 343 / 2003 / 44. `native` gives 11 /
  29 / 68 and 132 / 712 / 20, bit-identical.
- The reference's four CPU threads throttle after ~20 s on this fanless box (ru-long 8478 → ~9850,
  dots 8929 → ~11 000 ms per call). wavo's iGPU stays at 750 MHz with the CPU idle.
- wavo's iGPU buffers are 529 MiB (GigaAM) and 914 MiB (Parakeet) at any clip length.
- i915 cancels a GPU request 20 s after it was queued ("Fence expiration time out") while wgpu
  reports success. With the encoder in one submission (7.3 s on ru-long, 11.6 s on dots alone)
  wavo returned the previous call's output or nothing: the fixture tests passed one at a time
  but not six in parallel, and `wavo run parakeet-v2` on 5 minutes (60 s segments) printed
  nothing. Fixed by one submission per block and a completion mark (see Log).
- llvmpipe (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`, native, `-n 3`): ru 15.1 s,
  ru-short 41.2 s, jfk 61.9 s warm; load 6–19 s.

## Long audio: one pass, no chunking (2026-10-10, phase 7)

On the M2, `wavo run` and `transcribe-cli` once each per file under `/usr/bin/time -l`: process
wall time with load and decoding, peak footprint in MiB. Russian is `ru-long` (33.8 s, 52 words)
on repeat; English is `dots-full`, `death` and `whole-earth` back to back (one speech); both cut
to length, 16 kHz mono PCM16. Load average 2.5–3.1.

| Model | Length | Time, s | Peak footprint | Words |
|---|---|---|---|---|
| `gigaam-v3` (e2e-rnnt) | 1 min | 0.76 / 0.79 | 349 / 344 | 91 |
|  | 2 min | 1.58 | 406 | 170 |
|  | 3 min | 2.45 | 459 | 236 |
|  | 5 min | 5.20 / 5.41 | 573 / 420 | 254 |
|  | 10 min | 15.9 / 20.0 | 858 / 529 | 177 |
| `parakeet-v3` | 1 min | 1.11 / 1.75 | 821 / 917 | 141 |
|  | 5 min | 7.59 / 14.0 | 1132 / 1110 | 796 |
|  | 10 min | 22.5 / 43.9 | 1516 / 1293 | 1589 |

- The text is identical to the reference's at every length measured against it.
- GigaAM was trained on ≤ 25 s (the reference's `docs/input-limits.md`: soft window, it warns and
  proceeds; wavo says nothing). Past ~1 minute it drops words: about 92 / 185 / 277 / 462 / 923
  are spoken, 91 / 170 / 236 / 254 / 177 come out. Long Russian audio needs chunking.
- Parakeet stays coherent at 10 minutes (7500 encoder frames, full attention), but it drops
  sentences (see the split below).
- Time grows faster than length (attention is quadratic): ×5 length is ×6.8 for GigaAM and
  Parakeet, ×10 is ×21 and ×20.
- Peak footprint grows with length faster than the reference's: from 1 to 10 minutes +509 vs
  +185 MiB (GigaAM), +695 vs +376 MiB (Parakeet). Not investigated.

## Long audio: split at pauses (2026-10-10, phase 9)

`wavo run` with its default split: segments of up to 25 s for GigaAM (`max_audio_ms`) and 60 s
for Parakeet, cut at the quietest 300 ms. The files and protocol are the same as for the one-pass
table above, plus 60 minutes (both sources on repeat). Load average 2.6–3.5. Words are counted in
lowercase without punctuation (a hyphenated word is one word). For Russian, the brackets hold the
count of the fixture's words on repeat, cut at the same length. English is compared with
`--segment 0` (one pass).

| Model | Length | Time, s | Peak footprint | Words |
|---|---|---|---|---|
| `gigaam-v3` (e2e-rnnt) | 1 min | 0.75 | 313 | 91 (91) |
|  | 2 min | 1.12 | 315 | 177 (178) |
|  | 3 min | 1.64 | 323 | 263 (263) |
|  | 5 min | 2.69 | 327 | 455 (455) |
|  | 10 min | 5.23 | 349 | 905 (905) |
|  | 60 min | 31.0 | 529 | 5417 (5418) |
| `parakeet-v3` | 1 min | 1.34 | 822 | one segment |
|  | 2 min | 1.87 | 825 | 305, 1 differs |
|  | 3 min | 3.17 | 830 | 492; one pass 454 |
|  | 5 min | 4.57 | 844 | 809, 3 differ |
|  | 10 min | 8.91 | 865 | 1652; one pass 1622 |
|  | 60 min | 52.9 | 1052 | 9820 |

- GigaAM keeps every word: the only differences from the fixture text are in the last one or two
  words, which the end of the file cuts. `gigaam-v3-rnnt` (characters) at 10 minutes: 976 (977),
  the same. One pass gives 177 at 10 minutes.
- Parakeet in one pass loses text: at 3 minutes it stops 38 words before the end, at 10 minutes it
  skips a 29-word sentence. The split keeps both. The other differences are spelling ("17" /
  "seventeen", hyphens) or one word next to a cut, and the split is right as often as the one
  pass ("stewart", "lived", "drown"). V2 at 5 and 10 minutes: 2 and 5 words differ, the same kind.
- At 10 minutes, against the one pass in the same session (A/B/A): GigaAM 5.22–5.23 vs
  15.6–15.7 s and 349–351 vs 856–862 MiB (transcribe.cpp in one pass 20.0 s, 529 MiB); Parakeet
  8.9 vs 22.9–24.9 s and 865 vs 1513 MiB (transcribe.cpp 43.9 s, 1293 MiB).
- Time is linear in length: 10 → 60 minutes is ×5.9 for both, ~0.52 s (GigaAM) and ~0.88 s
  (Parakeet) per minute of audio. Peak footprint is flat apart from the decoded audio (f32,
  ~230 MB per hour): +180 MiB from 10 to 60 minutes.
- Segment length for Parakeet, `--segment` 30 / 60 / 120 on the 10-minute file (two runs each):
  8.57–8.75 / 8.85–9.03 / 9.81 s (the second 120 s run hit a load burst, 13.6 s), 830 / 861–865
  / 954–955 MiB. All three keep the sentence the one pass skips; 120 s skips another one (22
  words). Over 60 minutes, 30 s twice inserts a phrase nobody says ("that was a very good thing")
  where 60 s doesn't. 60 s is the default.

## Real recordings vs transcribe.cpp (2026-10-10, phase 10)

The person's own dictation, as a voice-typing app (Handy) saved it on this M2: 78,791 WAVs, all
16 kHz mono PCM16, 121.5 hours, mostly Russian (3% of the sample came out of Parakeet V3 in Latin
letters only). Length: median 3.75 s, p10 1.26 s, p90 11.9 s, p99 27 s, longest 121 s; 24% under
2 s, 37% 2–5 s, 24% 5–10 s, 13% 10–25 s, 1.3% over 25 s. The recordings are private: only
aggregates are kept here.

Sample: 2,923 recordings (8.63 h) drawn at random per length (500 under 2 s, 700 of 2–5 s, 700 of
5–10 s, 600 of 10–25 s, 400 of 25–60 s) plus all 23 of 60 s or more, in shuffled order. Protocol
as `make compare`: one model load per process and one pass per file (`examples/batch.rs` calls
`Model::transcribe`; `transcribe-cli --batch LIST --batch-jsonl` with default flags calls
`transcribe_run`), wavo, reference, wavo (A/B/A) on the same list in the same order with the files
in the page cache, all of it under one hold of the GPU lock. A file's time: wavo the wall time of
`transcribe`, the reference its `mel_ms + encode_ms + decode_ms`. That sum leaves out 2.1–2.6 ms
of the reference's call on GigaAM and ~6 ms on Parakeet (`transcribe-bench` `wall_ms − total_ms`
on these recordings and `jfk`); the process wall time includes it. GigaAM ran twice: A/B/A while
other agents built and benchmarked (load average 2–16), then B/A/B on a quiet machine (1.2–1.8).

| Model | Files | Same text | WER between the two | Files that differ |
|---|---|---|---|---|
| `gigaam-v3` (e2e-rnnt) | 2,923 | 99.86% | 0.004% (3 of 77,561 words) | 4: 3 one-word, 1 punctuation only |
| `parakeet-v3` | 2,923 | 98.9% | 0.03% (23 of 77,416) | 32: 20 one word or token, 12 punctuation or case only |
| `parakeet-v2` | 87 in Latin letters | 97.7% | 0.10% (3 of 2,897) | 2: one word each |

| Model | Run | Compute, s | Process wall, s | RTF median / p90 / max | Load, ms | Peak footprint, MiB |
|---|---|---|---|---|---|---|
| `gigaam-v3` | A/B/A, busy | 296.5, 302.4 / 309.4 | 302.9, 309.4 / 322.3 | 0.0097 / 0.0179 / 0.040, 0.0100 / 0.0182 / 0.154 vs 0.0099 / 0.0157 / 0.054 | 216, 194 / 206 | 452, 459 / 421 |
| `gigaam-v3` | B/A/B, quiet | 289.1 / 293.5, 292.6 | 292.6 / 301.9, 301.0 | 0.0095 / 0.0176 / 0.022 vs 0.0094 / 0.0151 / 0.019 | 83 / 165, 128 | 443 / 401 |
| `parakeet-v3` | A/B/A | 524.0, 511.5 / 814.6 | 530.9, 519.2 / 837.0 | 0.0184 / 0.0371 / 0.089, 0.0180 / 0.0357 / 0.129 vs 0.0261 / 0.0407 / 0.587 | 432, 412 / 344 | 975, 989 / 1067 |
| `parakeet-v2` | A/B/A, 87 files | 18.8, 17.6 / 23.8 | 19.4, 17.9 / 24.7 | 0.0171 / 0.0394 / 0.055 vs 0.0238 / 0.0371 / 0.055 | 408, 152 / 269 | 886, 884 / 925 |

Each cell is wavo / reference; a comma separates the two runs of the same engine. By length, mean
ms per file and the median of the per-file ratio wavo / reference (GigaAM from the quiet B/A/B,
Parakeet V3 from its A/B/A; stage sums as above), and the whole corpus estimated from the bucket
means:

| Length | Share of corpus | `gigaam-v3` ms | Ratio | `parakeet-v3` ms | Ratio |
|---|---|---|---|---|---|
| < 2 s | 24.3% | 25.5 / 22.0 | 1.16 | 53.1 / 61.0 | 0.93 |
| 2–5 s | 37.4% | 37.3 / 35.1 | 1.07 | 77.3 / 104.7 | 0.80 |
| 5–10 s | 24.1% | 65.2 / 64.1 | 1.01 | 126.0 / 186.5 | 0.72 |
| 10–25 s | 12.9% | 126.0 / 129.4 | 0.98 | 227.2 / 355.2 | 0.67 |
| 25–60 s | 1.3% | 278.9 / 293.6 | 0.95 | 477.8 / 780.8 | 0.65 |
| ≥ 60 s | 23 files | 753.7 / 783.3 | 0.96 | 1207 / 2386 | 0.61 |
| All 78,791, estimated | | 73.3 / 71.6 min | | 141.6 / 203.8 min | |

With the model loaded per recording (40 recordings drawn from the whole corpus, median 3.3 s; a
fresh `wavo run --segment 0` and a fresh `transcribe-cli` per file, alternating, two rounds;
process wall time): GigaAM median 136 / 188 ms (mean 156 / 206), Parakeet V3 278 / 429 ms (mean
300 / 470); wavo was faster in 80 of 80 runs for each.

- Text: the two engines agree on 99.86% (GigaAM), 98.9% (Parakeet V3) and 97.7% (Parakeet V2) of
  the recordings, and both return an empty text on the same 8 / 12 / 6. No errors or crashes in
  either. Each engine gives the same text on every rerun. Every differing file starts differing at
  one decision where wavo's top two logits (a token against another token or blank, or two TDT
  durations) are within 0.0001–0.032 of each other, 28 of 38 under 0.005, while the median margin
  per file is 2–11 (measured with a temporary print in the decoders, not kept): near-ties that
  float rounding tips one way or the other. The texts then differ in one or two places per file
  at most. Of 300 random recordings with token times (`wavo run --json` against `transcribe-cli
  --timestamps token`): GigaAM 300 same text and 296 also the same tokens and start times,
  Parakeet V3 298 and 289; the rest of the same-text ones differ only in some start times. No
  sign of a bug.
- GigaAM past its 25 s window (423 of the sample, 1.3% of the corpus; both engines run it in one
  pass and only the reference warns): 2 of its 4 differences, 99.5% the same text.
- GigaAM speed: on par over the whole sample (compute 1.2–1.5% below the reference on the quiet
  machine, 2–4% on the busy one, where the reference's CPU decoder lost more; process wall time
  3–6% below). Per file it is slower on short recordings and faster on long ones: by stage sums
  16% slower under 2 s and 7% at 2–5 s, 2–5% faster from 10 s. With the reference's 2.3 ms of
  call overhead added, 5% slower under 2 s, on par at 2–5 s, 3–6% faster from 5 s. For the
  corpus's mix of lengths that is a tie (73.3 against 71.6 min by stage sums, 74.6 with the
  overhead added). A linear fit over the recordings under 10 s: wavo 13.6 ms + 7.4 ms per second
  of audio, the reference 10.3 ms + 7.7 ms (+ ~2.3 ms outside its stages). So wavo's fixed cost
  per call is ~1 ms higher; see "Not tried yet".
- Parakeet V3: 36–37% less compute (524 / 512 against 815 s) and 37–38% less process wall time,
  faster in every length bucket and on 72% (under 2 s) to 100% (over 60 s) of the files; 142
  against 204 minutes for the corpus. V2 on the 87 English-looking recordings: 21–26% less
  compute, but slower under 2 s (2.0 against 1.8 s for 40 files): its reference decoder has no
  8193-way softmax.
- Peak footprint over the batch: Parakeet 975–989 against 1067 MiB, GigaAM 443–459 against
  401–421 MiB. wavo keeps the arena of the longest input so far (here 121 s; see "Long audio" for
  how it grows), while the reference allocates per call.
- Load in the A/B/A runs came first after other GPU work and the model file had left the page
  cache; with warm caches a fresh wavo process loads in 78–84 (GigaAM) and 162–170 ms (Parakeet
  V3), as in the benchmark above.
- Handy's own stored transcripts (no ground truth, and the database names no engine): since
  2026-09-28, when the GigaAM e2e-rnnt GGUF entered the Hugging Face cache (Handy's settings now
  name it), 299 of the 309 sampled recordings have Handy text equal to wavo's and the reference's,
  and the other 10 differ from both. Before that, 6–36% per month equal GigaAM's and 6–13%
  Parakeet V3's text (WER against Handy 7.5–17% and 11–15%): other engines (Handy's model folder
  holds int8 ONNX GigaAM v3 and Parakeet V2/V3, Whisper large-v3-turbo and Moonshine).

## Targets: transcribe.cpp Metal (commit 5bb2deb, same Q8_0 GGUF)

The reference side of the benchmark above. Earlier measurements (busier machine, load average
3–5): GigaAM e2e-rnnt warm 44–45 / 102–103 / 312–326 ms, load 0.13–0.17 s, first call 48–73 /
103–124 / 319–353 ms, peak 322 / 324 / 334 MiB; Parakeet warm 209–213 / 799–821 / 95–97 ms, load
0.26–0.27 s, first call 262–284 / 720–926 / 110 ms, peak 820 / 834 / 815 MiB.

The same C++ build on F32-expanded weights: GigaAM 45 / 110 / 368 ms. So Q8 storage itself buys
0–14% warm speed. Its big wins are memory and load time.

Requirement: GigaAM e2e-rnnt warm must beat these numbers; other models at least match them. Load,
first call and memory should beat them too.

The reference is not an F32 pipeline. Its Metal matmul (`kernel_mul_mm_q8_0_f32`) dequantizes
Q8_0 weights to half, casts activations to half, multiplies with `simdgroup_half8x8` and
accumulates in F32.

## Current state: GigaAM v3 e2e-rnnt (phase 2 done)

Numbers: the benchmark table above (warm 2–4% faster than the reference, first call on par or
below, load ~40% and peak footprint 12–29 MiB below). The first run after a shader change loads
in 0.2–0.4 s while Metal compiles and caches the new shaders.

How it runs: the GGUF header is read at load and each tensor is streamed from the file straight
into weight buffers that are mapped on unified memory (no staging copy); weights stay Q8_0/F16
on the GPU. Load ends with an encoder run on 8 silent frames to pay the GPU's first use of
pipelines and weights. Per call: CPU log-mel (rustfft, one frame at a time), then the whole
encoder as one compute pass over an arena sized for the longest input so far, with bind groups
cached per dispatch. Fast kernels (cooperative-matrix GEMM with Q8_0/F16 dequantized while
staging and rounded to f16 like the reference's, barrier-free flash attention, one subgroup per
row for LayerNorm and the conv module) each turn on after a probe dispatch; portable WGSL
kernels run otherwise. The head's linear layer is the last GEMM: the joint's encoder projection
for RNN-T, the logits (rows padded to 64) for CTC. The RNN-T loop (LSTM, joint, argmax) runs on
one CPU thread with NEON tiles that score two frames per pass over the joint weights; CTC is an
argmax and collapse per frame.

### Profile (2026-10-09, GPU timestamps per dispatch, warm, ms per call)

Measured with a temporary tool (removed): each dispatch in its own compute pass with begin/end
timestamps, which inflates the GPU span ~5%; CPU timers in the normal single-pass mode.

| | ru (T=113) | ru-long (T=845) |
|---|---|---|
| GPU kernels total | 42.2 | 306 |
| FFN down, 768←3072 Q8_0 + residual (32×) | 12.3 | 71.6 |
| FFN up, 3072←768 Q8_0 + SiLU (32×) | 11.2 | 71.0 |
| q·k projection, 1536←768 Q8_0 (16×) | 4.2 | 26.8 |
| conv pointwise1, 1536←768 F16 (16×) | 2.9 | 16.8 |
| v projection, 768←768 Q8_0 (16×) | 2.4 | 13.6 |
| attention out, 768←768 Q8_0 + residual (16×) | 1.7 | 9.5 |
| conv pointwise2, 768←768 F16 + residual (16×) | 1.6 | 8.8 |
| pre-encode convs + joint projection | 0.6 | 3.2 |
| attention (16×) | 2.3 | 73.7 |
| LayerNorm 64×, conv_glu 16×, LayerNorm+RoPE 16×, im2col 2× | 2.6 | 11.2 |
| CPU: mel / encode + submit / GPU wait / readback | 1.0 / 0.3 / 40 / 0.01 | 7.7 / 0.4 / 285 / 0.05 |
| CPU: decoder | 5.0 (ref 2.6) | 37.6 (ref 19) |
| First call: GPU wait | 83 (+43) | 324 (+40) |

- GEMM is 88% of GPU time on ru and 72% on ru-long; attention is 24% on ru-long.
- FFN GEMMs run at ~1.8 TFLOP/s on ru-long but ~0.7 TFLOP/s on ru: at M=113 FFN down has only
  48 workgroups (N/64 × ⌈M/32⌉) for a 3072-long K loop.
- The same 1536←768 shape takes 26.8 ms with Q8_0 weights and 16.8 ms with F16: dequantizing
  while staging W costs ~60% on top.
- Recording and submitting the 261 dispatches costs < 0.5 ms; bind groups are reused.
- At that point the first call's extra ~40 ms was GPU time: wgpu ran the weight uploads (staging
  → GPU, ~285 MB) with the first submit, not at load. Peak memory held the whole GGUF read into
  memory, the Q8_0 repack copies and wgpu's staging buffers next to the GPU weights. Both are
  fixed; see the Log.

## Current state: other GigaAM v3 variants (phase 3)

Numbers: the benchmark table above (warm 1–9% faster than the reference, the CTC models most).
A cold first read of a new GGUF took wavo 0.19–0.47 s to load. The CTC variants have no decoder
loop; their reference reads the encoder output back and runs the head and a log-softmax on the CPU.

## Current state: Parakeet TDT V2 (phase 5 done)

Numbers: the benchmark table above (end of phase 6: warm 14–34% faster than the reference, first
call 24–38% below, load ~40% and peak footprint 66–82 MiB below). Phase 4 ended at warm 181 /
756–763 / 94 ms and peak 810 / 877 / 800 MiB (jfk / dots / jobs-silence), phase 5 at 149 / 503 /
83 ms and 765 / 790 / 757 MiB; see the Log for each step.

Where the time goes at the end of phase 5 (the temporary profiling tool, warm, ms, jfk / dots):
GPU span 140 / 420 (kernels 145 / 456 with overlap): FFN GEMMs 70 / 204, the other GEMMs 54 /
161 (`[q | k | v]` 14 / 42, `linear_pos` 12 / 55 with overlap, attention out 13 / 28, pointwise
14 / 36), attention 10 / 64, LayerNorm 4.6 / 8.5, conv_glu 2.1 / 7.4, subsampling and joint
projection 3.6 / 10.6. CPU: mel 2.7 / 8.8, decoder 20 / 113 (LSTM on two threads; 13–15 / 71–78
with phase 6's Q8_0 decoder). Not done: conv_glu's taps as [tap][ch] (≤ 2% of GPU time, shared
with GigaAM) and GEMM tile shapes at M = 138 (GigaAM's tile experiments gained nothing); the
encoder is far ahead of the reference.

How it runs: the shared Conformer block (`src/conformer.rs`) with relative attention: one GEMM for
`[q | k | v]` whose epilogue writes `[q + u | k | v | q + v]` (biases `pos_bias_u`, 0, 0 and q again
with `pos_bias_v`), `P = linear_pos(pos_emb)` per block, then flash attention that scores each
16-query × 64-key block against the 80 positions it needs and skews those scores onto the keys in
shared memory (see Log). The sinusoidal table is cached on the CPU for the longest T so far and its
2T + 15 rows (16 extra positions for the kernel's partial tiles) are written per call. Subsampling:
conv0 + ReLU fused into the first depthwise conv (conv0's output is never stored), pointwise convs
and the projection through the GEMM, a flatten kernel to `[t][c·16 + f]`; conv0 to conv5 run in 8
chunks of output rows, so their outputs fit in the blocks' buffers. The joint's encoder projection
is the last GEMM; the TDT loop and two LSTM layers run on the CPU over Q8_0 weights on four
threads (see Parakeet TDT V3). BatchNorm is folded into a per-channel affine at load.

### Profile at the start of phase 5 (2026-10-10, GPU timestamps per dispatch, warm, ms per call)

Same temporary tool as GigaAM's (removed). Passes of consecutive dispatches that don't depend on
each other overlap, so the kernels sum to more than the span: `linear_pos` right after the qkv
GEMM is charged 46.9 ms on dots, 31.6 ms when recorded before it (the rest moves < 2%).

| | jfk (T=138) | dots (T=442) |
|---|---|---|
| GPU kernels total / span | 166 / 165 | 674 / 638 |
| FFN up, 4096←1024 Q8_0 + SiLU (48×) | 35.1 | 95.6 |
| FFN down, 1024←4096 Q8_0 + residual (48×) | 34.2 | 96.5 |
| `[q + u \| k \| v \| q + v]`, 4096←1024 Q8_0 (24×) | 17.3 | 48.7 |
| `linear_pos`, 1024←1024 Q8_0 on 2T − 1 rows (24×) | 11.0 | 31.6–46.9 |
| conv pointwise1, 2048←1024 F16 (24×) | 9.1 | 23.5 |
| attention out, 1024←1024 Q8_0 + residual (24×) | 4.6 | 13.1 |
| conv pointwise2, 1024←1024 F16 + residual (24×) | 4.2 | 12.3 |
| attention: position scores / scores / values / softmax (24× each) | 15.4 / 14.9 / 6.5 / 1.2 | 150.4 / 98.2 / 48.1 / 3.7 |
| subsampling: conv0 + conv2 (`first_depthwise`) / the rest | 5.7 / 1.7 | 18.9 / 5.3 |
| LayerNorm 120×, conv_glu 24× | 3.2 / 2.1 | 5.3 / 6.9 |
| CPU: mel / encode + submit / GPU wait / readback | 2.7 / 0.5 / 153 / 0.01 | 8.8 / 0.6 / 618–630 / 0.03 |
| CPU: LSTM steps (tokens) / joint (decisions) | 21.2 (33) / 2.4 (43) | 116 (194) / 11 (203) |

- Attention is 23% of GPU time on jfk and 45% on dots, ~300 ms. Its kernels are scalar 16×16
  tiles through memory; the position scores cover all 2T − 1 positions per query.
- GEMMs run at 1.5–1.6 TFLOP/s at T = 138 and 1.7–1.9 TFLOP/s at T = 442, as GigaAM's.
- The stacked GEMM computes q twice: a quarter of it (~4 / 12 ms) and 25.5 MiB of weights.
- `first_depthwise` does ~0.2 / 0.6 GFLOP in 5.7 / 18.9 ms: one workgroup per pixel, conv0
  recomputed per tap, per-channel weights read 9 floats apart.
- The LSTM step (two layers of Wx and Wh 2560×640 in F32 plus the predictor projection, 28 MB of
  weights) takes 0.6 ms per token, ~47 GB/s: memory-bound. The joint takes 0.05 ms per decision.
  jobs-silence: mel 1.3, GPU 91, decoder 1.6 ms.
- Memory: GPU weights 711 MiB, decoder weights in F32 32 MiB. The arena (frames rounded up to 64:
  jfk 192, dots 448) is 35 / 93 MB; on dots `h` and `qk` are 29.4 MB each, sized by the
  subsampling's [2T][32][256] (4× what the blocks need), `ps` 12.8, `s` 6.4, `yr` + `pos` 7.3 MB.
- First call over warm: +8 / +0–9 / +4 ms.

## Current state: Parakeet TDT V3 (phase 6 done)

Numbers: the benchmark table above (warm 28–35% faster than the reference, first call 43–51%
below, load ~40% and peak footprint ~120 MiB below). Encoder, frontend and TDT settings are V2's
(identical metadata); the vocabulary has 8193 pieces, so the joint has 8198 outputs (21 MB per
decision in F32, V2 2.6 MB) and the embedding table 8193 rows. The reference keeps both in F32
and runs its decoder as ggml CPU graphs on up to 8 threads, plus a softmax over 8193 tokens per
emission for its confidence: `decode_ms` 72–118 on these clips.

How the decoder runs (both Parakeet models): every decoder weight stays Q8_0 (`cpu::Q8`, int8
values and an f32 scale per 32) and is dequantized in NEON registers; `q·d` is exact in f32 and
the lanes, fused multiply-adds and pairwise sum are `dot`'s, so the results are bit-identical to
the F32 path. A step's LSTM (6.6 MB) and joint (5.2 MB) plus scales fit in the performance
cores' 16 MB L2. Each product goes out in blocks of 64 rows to four scoped threads that take the
next block when free (`matvecs`), so a thread on an efficiency core takes fewer; a layer's
`Wx·x` and `Wh·h` share one queue. Decoder (direct timer, median of 18 calls) jfk / ru-short /
uk-short: 21 / 31–36 / 29–31 ms, from 76 / 68 / 70 ms in F32. Per decision the joint takes
~0.18 ms (F32 0.75–0.9), per token the predictor ~0.32 ms (F32 0.9–1.1: the joint's 21 MB evicted
the LSTM's weights from L2).

## Research history: `research` branch (all weights expanded to F32)

| Model | Warm | vs C++ | Load | RSS after load | Peak footprint |
|---|---|---|---|---|---|
| GigaAM | 56.5 / 118 / 374.5 ms | 1.18–1.29× slower | 0.65 s | ~0.9 GB | ~5.0 GB |
| Parakeet V2 | jfk 200.5, dots 644 ms | on par / 21% faster | 1.8 s | ~2.4 GB | ~7.8 GB |

GigaAM load breakdown (ms): GGUF read 25, dequant to F32 79, GPU upload 227, weight packing 209,
other 86. Parakeet: upload 633, packing 590, dequant 217 of 1755. All of this comes from expanding
to F32; the GGUF files are only 0.27 / 0.73 GB.

### What worked (research: naive 621 / 1389 / 4311 ms → 56.5 / 118 / 374.5 ms)

1. Cooperative-matrix GEMM (Metal simdgroup 8×8 via wgpu `EXPERIMENTAL_COOPERATIVE_MATRIX`,
   F32): 9–15× per GEMM over a scalar tiled GEMM. 128 threads = 4 subgroups × 32; output tile
   32×32, K step 16. Larger tiles and larger K were slower. Peak measured ~1.6 TFLOP/s.
2. `PipelineCompilationOptions { zero_initialize_workgroup_memory: false }`: Naga otherwise clears
   shared memory from a single lane.
3. `gemm_fast`: output tile M32×N64, K16, vec4 loads, 8 accumulators per subgroup: another
   1.17–1.96× per GEMM (largest gain at small M).
4. Prepacking weights at load into 8×8 fragment order (no B transpose per tile): 1.02–1.26× per
   GEMM, ~8% end to end, +0.21 s load on CPU.
5. Subgroup reductions: LayerNorm 2.5–2.9×, softmax 1.4–14× (microbenchmarks).
6. One command buffer per Conformer block (17 submits per inference) instead of one per op.
7. CPU RNN-T decoder with NEON/FMA matvec; predictor output cached across blanks.

### Profile of the research state (GigaAM; GPU timestamps, ms per inference, ru / ru-short / ru-long)

- GEMM 48 / 102 / 348 = 68–79% of GPU time. FFN up+down alone 29 / 59 / 215.
- Attention (scores + softmax + values) 4 / 13 / 104: grows ~T², 20% at 34 s.
- Elementwise passes (SiLU, residual adds, GLU, RoPE) ~5 / 12 / 43: can be fused into GEMM
  epilogues.
- GPU idle gaps inside the encoder: 12 / 23 / 77 ms of a 69 / 147 / 562 ms span (~15%).
- CPU enqueue 21–40 ms per inference: 470 fresh output buffers, 470 uniform buffers and bind
  groups, 17 submits. 0.25 / 0.7 / 3.2 GB of buffers allocated per inference: this is the 5 GB
  peak footprint.
- CPU decoder 7 / 19 / 52 ms (10–14% of wall). Frontend 1–11 ms (ignore).

### Tried and rejected in research

Experiments 1–3 ran on a busy machine (load average up to 30, identical controls varying 2–3×).
Their verdicts are not reliable, so they may be retried.

1. CPU weight packing in local 8×8 order to speed up load: GigaAM −26%, Parakeet unclear.
   Moot once weights stay Q8 on the GPU.
2. Caching 128 small parameter uniform buffers: no visible gain. A full activation arena was never
   tried.
3. Decoder: batched encoder projection (NEON 4×4) plus 32-frame joint lookahead (what C++ does):
   inconclusive.
4. Fused online-softmax attention, coop matrices, 16 queries × 32 keys per workgroup (clean
   measurement): −28% peak memory, +12.6% time on ru-long. Rejected. A larger query tile was not
   tried.
5. Whisper: compensated (Neumaier) summation in GEMM/LayerNorm to bit-match the CPU reference.
   Correct but several times slower. Never do this; see agents.md → Correctness.

## Generic GPU lessons (from the Whisper work)

- Bound in-flight work: queuing all 32 encoder blocks at once kept every intermediate alive
  (22.7 GB footprint). Waiting every 4 blocks cut it to 9 GB and was faster.
- Dynamic indexing of private arrays in WGSL is slow on Metal. Explicitly unrolled scalar
  accumulators made a GEMM 2× faster.
- M=1 (decoder step, output head) needs a dedicated matvec kernel. A 32-row GEMM tile with one
  useful row was 2× slower (16.6 → 8.6 ms on a 51866×1280 head).

## Naga / wgpu quirks (30.0.1)

- `enable subgroups;` is rejected even when `SUBGROUP` is supported; just use subgroup builtins.
- Cooperative matrix splat `CM(0.0)` is rejected: use a zero-initialized `var c: CM`.
- Cooperative ops inside a workgroup-position-dependent branch fail uniformity validation. Keep
  them and barriers outside bounds branches; zero-fill partial tiles instead.
- Cooperative matrix addition compiles in Naga but emits Metal ops that don't exist.
- Verify cooperative support by running a tiny probe dispatch with the real pipeline. Adapter
  subgroup min/max (M2 reports 4..64) is not enough.
- Naga sizes a module's immediates (`var<immediate>`) from the first immediate variable it finds,
  so a second, larger struct in the same module overruns: use one params struct per module.
- A dispatch must set every immediate byte its entry point reads, but bytes it doesn't read may
  stay unset.
- Workgroup arrays sized by override constants (`array<f32, N>` with `override N: u32 = ...`)
  and `@workgroup_size(32 * GROUPS)` work: one source gives pipelines with different shared
  memory layouts.

## Where to look for kernel ideas

- `3rd/transcribe.cpp/ggml/src/ggml-metal/` is what the reference runs on Apple Silicon:
  `kernels/mul_mm.metal` (matmul), `kernels/mul_mv.metal` (matrix-vector), `kernels/fa.metal`
  (flash attention), `ggml-metal-fusion.cpp` (which ops it fuses), `ggml-metal-tuning.cpp`
  (per-device parameters).
- `3rd/transcribe.cpp/ggml/src/ggml-webgpu/wgsl-shaders/` is ggml's WebGPU backend, written in
  WGSL like ours, with Q8_0 `mul_mat`, `mul_mat_vec` and flash attention. transcribe.cpp doesn't
  use this backend, but its shaders can be adapted directly (research did this for the Whisper
  output head).

## Not tried yet (GigaAM, after phase 2)

The phase 1 list (GEMM tile shape, split-K, prefetched staging, attention blocks and rescale
skip, NEON decoder) and the research list are done or rejected; see the Log.

1. Attention: q fragments in registers in a kernel specialized for head_dim 48 (only constant
   loop bounds were tried). Its softmax (~17 of 50 ms on ru-long) still runs 32 subgroup
   reductions per 64-key block.
2. q·k and v projections as one GEMM (concatenated weights, one A buffer holding `yr | y`): one
   dispatch less per block and more workgroups at small M.
3. Q8_0 prepacked in fragment order: low priority, W traffic is not the GEMM's limit (see Log).
4. Decoder: `Wx · embed[token]` depends only on the token, so a per-call cache would skip half
   of the LSTM step (~68 µs) for repeated tokens.
5. Short recordings (see "Real recordings"): under 2 s (24% of real dictation) wavo's call costs
   ~1 ms (5%) more than the reference's wall time, at 2–5 s (37%) the same. GigaAM is meant to be
   faster everywhere. At T ≤ 64 frames (2.5 s) FFN down has 12–24 workgroups (N/64 × ⌈M/32⌉),
   under two per GPU core. Profile a 1–2 s clip first; candidates are split-K or a 16-row tile at
   small M, item 2, and fixed CPU costs per call.

## Log

Add entries here, newest first: date, model, idea, before → after (median, A/B/A), verdict, why.

- 2026-10-10, GigaAM e2e-rnnt, Parakeet TDT V3 (and V2 on 87 recordings), 2,923 of the person's
  dictation recordings against transcribe.cpp's batch mode (`make compare`): text the same in
  99.86% / 98.9%, every difference from a near-tie; Parakeet V3 36–37% less compute, GigaAM on
  par overall (slower per file under 2 s, faster from 5 s). See "Real recordings" above.

- 2026-10-10, all models, GPU work that fails or that a driver cancels is an error, and the
  portable path submits each Conformer block on its own. i915 cancels a request 20 s after it
  was queued and wgpu reports success, so every submission now ends with a `mark` dispatch that
  writes the call's number to its own slot, read back with the output; wgpu's uncaptured and
  device-lost errors are kept too. Per-block submissions only helped once each waited for the
  one before (i915 starts the clock when a request is queued). N100: `make test` in parallel
  0 → 6 of 6, `wavo run parakeet-v2` on 5 minutes 0 → 808 words, a 5-minute call in one pass an
  error instead of stale text. M2, per block for every GPU (A/B/A/B/A, warm median): ru 43.4–43.8
  → 44.0–44.2, jfk 141.6–143.2 → 142.9–144.2, dots 452.8–453.1 → 455.1–456.5 ms, ru-long equal;
  so the fast path stays one submission, and against the code before the change ru 43.4–44.0 →
  43.5–43.7, ru-long 297.6–298.2 → 298.0–298.7, jfk 142.5–143.8 → 141.9–142.3, dots 453.6–456.7
  → 453.9–456.4 ms, first call and load within noise. Kept.

- 2026-10-10, Linux N100 (ANV), `MemoryHints::MemoryUsage` in the device descriptor: iGPU buffers
  529 → 324 MiB (GigaAM), 914 → 777 MiB (Parakeet V2); ru 828 → 828, jfk 3645 → 3646 ms (one run
  each). Not kept yet: needs A/B/A, the M2 and the fixtures (plan `20261010-linux.md`).
- 2026-10-10, Linux N100 (ANV), `-C target-cpu=native`: CPU decoder 13–15× faster on GigaAM,
  2.6–2.8× on Parakeet V2, fixtures exact; warm ru-short 2536 → 2133 ms, dots 13628 → 12337 ms.
  Not kept as a build flag: runtime-dispatched AVX2 tiles instead (plan `20261010-linux.md`).

- 2026-10-10, GigaAM e2e-rnnt and Parakeet TDT V3, `wavo run` splitting long audio at pauses
  (≤ 25 / 60 s segments), 1 to 60 minutes: 10 minutes 15.6 → 5.2 s and 22.9 → 8.9 s, peak 856 →
  349 and 1513 → 865 MiB, GigaAM 177 → 905 words. Kept; see "Long audio: split at pauses" above.

- 2026-10-10, GigaAM e2e-rnnt and Parakeet TDT V3, 1 to 10 minutes in one pass through `wavo
  run`: see "Long audio" above.

- 2026-10-10, Parakeet TDT V2 and V3, end-of-phase-6 benchmark against `transcribe-bench`: see
  "Benchmark" above (the Parakeet rows).

- 2026-10-10, Parakeet TDT V3 and V2, decoder weights Q8_0 on four threads (direct timer, median
  of 18 calls; V3 jfk / ru-short / uk-short, V2 jfk / dots). F32 as in phase 5: V3 decode 76 / 68
  / 70 ms, of which the joint (8198 × 640) 35–42 / 33 / 31–34 ms, and the predictor 0.9–1.1 ms per
  token against V2's 0.52; V2 20.5 / 114 ms.
  - Every decoder weight Q8_0, dequantized in NEON registers, bit-identical (unit test against
    `dot`, fixtures exact): one thread per product V3 51 / 77 / 73, V2 27 / 150 ms; with each
    layer's `Wx·x` on a second thread as in phase 5, V3 41 / 61 / 58, V2 18 / 100 ms. The kernel
    alone (8200 × 640 rows): 486 µs, ~10.8 G weights/s, near its limit of 19 NEON ops per 16
    weights.
  - Rows split evenly over scoped threads: the kernel alone on 2 / 3 / 4 / 6 / 8 threads 265 /
    194–200 / 159–163 / 211 / 181–183 µs (past four the efficiency cores set the pace; a scoped
    spawn costs 7–15 µs). In place the V3 jfk joint on four threads took 8–17 ms from run to run.
  - Blocks of 64 rows handed to four threads as they come free (a locked chunk iterator; a layer's
    two products share it): the kernel alone 149–152 µs; decode V3 21–25 / 31–36 / 29–35, V2 13–15
    / 71–78 ms. On 3, 6 and 8 threads V3 jfk 23.6 / 22.6 / 24.8, V2 dots 80.9 / 79.8 / 91.3 ms.
    Kept at four. The 640-row `joint.pred` on one thread or four: the same within noise, so
    `matvecs` always uses four.
  - Rejected: dequantizing each group of 4 rows into a buffer for the F32 `tile::<4, 1>` (no NEON
    kernel of its own, ~29 lines less): 1965 vs 486 µs per kernel call.
  - Warm A/B/A/B against de4fef2 (load average 2.4–10): V3 jfk 175.3 / 149.6 / 174.6 / 150.0,
    ru-short 197.1 / 160.0 / 197.3 / 160.6, uk-short 193.7 / 161.7 / 193.6 / 164.3 ms; V2 jfk
    150.3 / 144.3 / 149.7 / 143.7, dots 498.1 / 461.1 / 497.4 / 461.0, jobs-silence 84.8 / 84.1 /
    84.2 / 84.0 ms; GigaAM e2e-rnnt (its decoder unchanged) ru 45.2 / 45.0 / 45.5 / 44.9, ru-long
    302.1 / 309.1 / 299.3 / 299.0 ms. Peak footprint V3 808 → 762 (jfk), 806 → 761 MiB
    (uk-short); V2 763 → 741 (jfk), 794 → 766 MiB (dots). Kept: +95 lines in `src/cpu.rs`.
- 2026-10-10, Parakeet TDT V3, first run on the V2 code path: fixtures `jfk`, `ru-short` and
  `uk-short` exact with no code change. Warm 175 / 198 / 196 ms (reference 216 / 251 / 240 ms in a
  quick check), peak 804 vs 884 MiB.

- 2026-10-10, all models, the same benchmark on an Apple M1 (macOS 15.8.1) with the M2-built
  binaries: see "Second machine" under "Benchmark".

- 2026-10-10, all models, end-of-phase-5 benchmark against `transcribe-bench`: see "Benchmark"
  above. Re-profile of Parakeet before it: see "Current state: Parakeet TDT V2"; conv_glu's
  [tap][ch] taps and GEMM tile shapes not tried (small share, encoder far ahead).

- 2026-10-10, Parakeet TDT V2 and GigaAM, simplification (core + parakeet 3171 → 3103 lines): one
  LayerNorm kernel whose rotary output runs behind a uniform head_dim check (portable and fast;
  without it the table binding aliases x and `yr` is a 4-byte spare buffer, as WebGPU forbids
  aliased writable bindings), one `Pass::depthwise` with an optional
  first conv, `Norm` and `Depthwise` as one `Channels`, the decoder reading the joint's rows at
  the GEMM's padded width. Fixtures exact. Against dc832a2 (load average 4–6), Parakeet A/B/A/B
  jfk 154.4 / 154.8 / 154.9 / 154.4, dots 515.1 / 518.9 / 525.7 / 527.8 (drifting up for both),
  jobs-silence 86.1 / 86.5 / 86.5 / 87.3 ms, peak 761 / 764, 790 / 789 MiB; GigaAM B/A/B/A ru
  45.0 / 45.2 / 45.3 / 44.9, ru-long 308.4 / 308.0 / 307.9 / 307.5 ms. With the rotary check
  inside the fast kernel's per-element loop GigaAM ru was +0.3 ms (45.3 / 44.9 / 45.0 / 44.8), so
  the rotary output got its own loop.

- 2026-10-10, Parakeet TDT V2, decoder (direct timer, median of 20 calls, dots / jfk; the F32 LSTM
  step streams 28 MB: 129 / 24.0 ms):
  - Each layer's `Wx·x` on a scoped thread while `Wh·h` runs on the caller (bit-identical; no
    waiting thread between steps): 113 / 20.4 ms. Warm A/B/A jfk 153.4 / 151.0 / 153.1, dots
    517.1 / 501.2 / 516.3, jobs-silence 83.7 / 83.6 / 83.9 ms; peak +0–3 MiB. Kept (6 lines).
  - Four threads (row halves of `Wx` and `Wh`): 122–130 / 22.3–22.8 ms. Rejected: three spawns
    per layer and one memory bus.
  - LSTM weights kept Q8_0 (7 MB, fits in L2) and dequantized in NEON registers (q·d is exact in
    f32, same FMA order as `dot`, fixtures exact): one thread 142 / 25.5 ms (compute-bound on
    widen, convert and scale), two threads 96.7 / 17.5 ms and −24 MiB. Rejected for ~40 lines of
    NEON against 16 / 3 ms more than the F32 threads; the next step if the decoder matters more.
    Dequantizing 16 rows at a time into a buffer for the F32 tiles instead: 129 / 23.3 ms with
    threads, rejected.
  - Layer 0's `Wx·embed[token]` cached per call (dots repeats 92 of its 194 tokens, jfk 11 of
    33), on top of the threads: warm jfk 149.6 / 148.6 / 149.8, dots 503.8 / 499.5 / 500.1 ms.
    Within noise, rejected.

- 2026-10-10, Parakeet TDT V2, Task 3 (memory) as a whole, c1ea4fb / after / c1ea4fb (load
  average 3–6, Activity Monitor's `sysmond` at ~85% of a core): warm jfk 161.2 / 153.3 / 161.6,
  dots 567.6 / 531.0 / 554.7, jobs-silence 90.7 / 85.8 / 90.7 ms; peak footprint 808 / 762 / 809,
  857 / 791 / 857, 796 / 760 / 795 MiB (reference 820 / 834 / 815). GigaAM e2e-rnnt A/B/A/B ru
  43.3 / 43.5 / 43.4 / 43.3, ru-long 296.5 / 296.8 / 296.5 / 297.1 ms, peak 322 / 323 MiB.
- 2026-10-10, Parakeet TDT V2, subsampling in 8 chunks of ⌈T/8⌉ output rows (always 8, empty ones
  dispatch nothing, so the cached bind groups match): per chunk conv0 + conv2 into rows 2·t0 − 1..
  of a chunk buffer (`yr`), conv3 into `qk`, conv5 into its rows of the whole output (`h`); `h` and
  `qk` shrink to the blocks' size. Spatial weights as [tap][ch] (coalesced). Peak 780 → 765,
  830 → 789, 771 → 758 MiB. `first_depthwise` (GPU timestamps, jfk / dots): 5.7 / 18.9 ms before,
  4.8 / 12.4 with [tap][ch] weights, 3.3 / 11.3 with the 7×7 input pixels in shared memory,
  0.88 / 3.06 with conv0's nine weights also held as scalars (they were reloaded per tap). Kept.
  One workgroup per output row (mel rows in shared memory, weights in private arrays): 5.1 / 14.7
  ms, rejected. Warm after vs before (q once): jfk 160.4 / 156.3 / 158.5, dots 529.5 / 521.4 /
  542.6, jobs-silence 87.8 / 84.9 / 85.8 ms.
- 2026-10-10, Parakeet TDT V2, q computed once: the qkv GEMM computes `[q | k | v]` and its
  epilogue writes q again with `pos_bias_v` after v (`Linear::dual`, a bias longer than the rows),
  exact as before. Peak 807 → 782, 860 → 833, 798 → 770 MiB (25.5 MiB of q weights). Warm A/B/A
  jfk 164.2 / 160.6 / 163.4, dots 551.9 / 541.2 / 559.5, jobs-silence 90.6 / 88.0 / 90.6 ms. Kept.

- 2026-10-10, Parakeet TDT V2 and GigaAM, one flash kernel for both attentions. With one layout
  (one subgroup, scores [16][80], accumulator [16][128]) GigaAM got slower: ru 43.1 → 44.6,
  ru-long 296 → 328.6 ms; with the accumulator at [16][64] ru-long took 305.7 ms, so shared
  memory per query (occupancy) is the cost. Kept: one source whose override constant `REL` sizes
  the shared arrays and workgroup per pipeline (four subgroups, [16][64] scores and accumulator
  for rotary, as before); GigaAM A/B/A ru 43.2 / 43.1 / 43.2 / 43.2, ru-long 296.1 / 295.8 /
  296.3 / 295.9 ms. The portable relative attention adds `(q + v)·P` inside `scores` instead of
  a `pos_scores` kernel and buffer.
- 2026-10-10, Parakeet TDT V2, flash attention with relative positions for head_dim 128: one
  32-lane subgroup per workgroup owns 16 queries and walks 64-key blocks; per block it scores its
  q + v rows against the 80 positions the block needs (two passes of 5 × 2 fragments, so 10% more
  than the 72 + 72 needed), stores them with a row stride of 81 so position c + 15 − r of row r
  lands on key c, loads them as the initial accumulators of the q + u · k scores, then the
  head_dim ≤ 64 kernel's online softmax and P·V with a [16][128] accumulator. P gets 16 leading
  rows (the GEMM computes 2T + 15 rows) and 64 spare rows so partial tiles stay in the buffer.
  Fixtures exact. Warm A/B/A jfk 180.5 / 160.3 / 180.1, dots 762.4 / 544.4 / 760.7, jobs-silence
  94.4 / 88.2 / 94.0 ms; peak footprint 813 / 807 / 810, 876 / 858 / 874, 801 / 797 / 797 MiB
  (no `ps` and `s`). Attention alone (dispatch recorded twice): +8.5 ms on jfk, +57 ms on dots,
  from ~38 / ~300 ms. Kept. Two subgroups of 16 queries per workgroup (a clamp for the last
  block): jfk 160.2 / 160.1 vs 159.8 / 159.8, dots within noise; rejected for the simpler one.
- 2026-10-10, Parakeet TDT V2, profile with GPU timestamps per dispatch kind (temporary tool,
  removed): see "Profile" under Parakeet above.

- 2026-10-10, Parakeet TDT V2, phase 4 baseline recorded in "Current state" and "Targets" above
  (fixtures exact on the first run, no reference roundings needed). Shared core for it (Conformer
  block, relative attention, conv_glu kernel size and affine mode, attention kernels in their own
  module): GigaAM e2e-rnnt warm unchanged, ru 43.1 / 43.3 / 43.2 ms and ru-long 296.1 / 296.3 /
  296.1 ms A/B/A, then 43.2 / 43.1 / 43.2 / 43.2 and 296.0 / 295.7 / 295.4 / 295.6 ms B/C/B/C.

- 2026-10-09, GigaAM variants, GEMM rounds dequantized W to f16, as the reference's
  `kernel_mul_mm` does. Without it e2e-ctc ru-short emits one token a frame late: at frame 116
  the reference has 13.4252 for the token and 13.4183 for blank, wavo 13.4093 and 13.4172 (mean
  |Δ logits| 0.0026). With a and W both rounded: 13.4250 vs 13.4171, mean 0.0016, all argmaxes
  equal. W only: all four models' fixtures exact (e2e-rnnt unchanged), warm e2e-rnnt ru 43.7 /
  43.5 vs 43.8 / 43.5 ms without, ru-long B/A/B 296.0 / 295.2 / 296.4 ms. Kept. Rounding a as
  well: +1.0 ms on ru (44.9 / 44.5 ms), no fixture needs it, rejected. Rounding with integer ops
  instead of `pack2x16float`: +2.4 ms (W) / +4 ms (both) on ru, rejected.
- 2026-10-09, GigaAM variants, e2e-ctc, rnnt and ctc as heads of the shared encoder: CTC logits
  as the last GEMM (257 or 34 rows padded to 64 at load: `Gpu::linear` pads any layer with zero
  rows, the encoder drops the padding columns at readback), greedy collapse on the CPU; the
  charwise RNN-T reuses the decoder (34 classes). Bench in "Current state" above.
- 2026-10-09, GigaAM, small encoder experiments on ru (encoder wall time with a temporary timer;
  the machine drifts by ±1 ms over minutes, so only back-to-back runs count). All rejected:
  - 48 extra one-workgroup im2col dispatches: 39.9–40.4 vs 39.8–40.5 ms, so a dispatch boundary
    costs little. (Earlier runs, 48 extra one-row LayerNorms +1.4 ms and 48 LayerNorms fewer
    −1.2 ms, can't be told apart from the drift.)
  - Each block's output norm fused with the next block's first norm (one kernel, the first
    output recomputed per element, −15 dispatches): A/B/A 38.8–38.9 / 38.9–39.3 / 38.9–39.1 ms.
  - LayerNorm statistics with eight loads issued before their sums: warm A/B/A 44.4 / 44.2 /
    44.1 ms. LayerNorm rows held in a 32-entry register array: encoder +1.5 ms (likely spilled).
  - GEMM skipping the MMAs of a subgroup's lower 8 rows when they are all past m (6% of the
    MMAs at M = 113): the fast GEMM no longer passes its probe (a branch on the subgroup id
    around cooperative-matrix calls).
- 2026-10-09, GigaAM, attention softmax with two lanes per row (each lane loops over 32 keys of
  one parity, one `subgroupShuffleXor` per reduction instead of 16 × 2 subgroup reductions):
  rows 66 floats apart (no bank conflicts, but rows not 16-byte aligned) 98.5 ms, 72 apart
  (aligned, 4-way conflicts) 55.5 ms; scores and accumulator column-major with 16-float columns
  (aligned and conflict-free, lanes r and r + 16 on one row) 56 ms; vs 50.5 ms (GPU timestamps,
  ru-long). Rejected.
  Cooperative-matrix loads and stores on rows that are not 16-byte aligned cost ~2×. Ablations
  of the kept kernel: without the softmax 33 ms, without P·V 37.6 ms, so scores ~20, softmax
  ~17 and P·V ~13 ms of the 50.5.
- 2026-10-09, GigaAM, frontend: one frame at a time in a reused buffer (no 8.6 MB spectrum
  allocation on ru-long, no div/mod per sample) and each mel filter's `dot` only over its
  nonzero bins widened to 16-bin chunks (same lanes, the skipped terms are exact zeros). Mel
  bit-identical on all fixture clips (temporary bitwise check). Frontend (direct timer) ru
  1.0 → 0.57, ru-short 2.5 → 1.43, ru-long 7.6 → 4.4 ms (the rest is the FFT ~3 ms and `ln`).
  Warm ru-long A/B/A 307.8 / 303.2 / 307.5 ms. Kept.
- 2026-10-09, GigaAM, GEMM limits (GPU timestamps, ru-long, timing-only ablations with wrong
  outputs; FFN down / FFN up / q·k, normally 72–74 / 70–72 / 26–28 ms): W loads all from one
  2 KB region 71.3 / 67.6 / 27.8; a loads all from one region 73.8 / 73.1 / 25.8; staging (loads
  and shared-memory stores) only on the first K step 60.9 / 61.2 / 22.8; MMAs only on the first
  K step 37.3 / 30.6 / 11.3. So device memory traffic costs nothing, staging ~15%, and the rest is
  the on-chip loop; FFN up runs at ~1.8 TFLOP/s, about half the M2's FP32 peak. A 64×64 tile
  (each subgroup 32×32, 16 accumulators, 0.5 fragment loads per MMA instead of 0.75): 73.3 /
  70.3 / 26.7 ms, no gain. Rejected. Prepacking W in fragment order (plan item) not tried: W
  traffic is not the limit.
- 2026-10-09, GigaAM, GEMM split-K for short clips (timing-only ablation, outputs wrong: FFN down
  dispatched with 4 K-slices as `wg.z`, no reduction): FFN down at M = 113 (ru) 11.7–12.4 vs
  12.3–12.6 ms (GPU timestamps); only at M = 2 (load warm-up) it drops, 13.8 → 9.6 ms. At
  M = 113 its 48 workgroups already run at the throughput of FFN up's 192 (~1.55 TFLOP/s on the
  128 padded rows), so it is not short of workgroups. Rejected without the reduction pass.
- 2026-10-09, GigaAM, GEMM register prefetch (the next K step's a and raw W words load into
  registers before this step's multiplies): FFN down on ru 12.4–12.7 vs 12.3–13.4 ms; ru-long
  A/B/A noise (A 308.9 / 297.5 / 311.0, B 313.2 / 298.6 / 306.4 ms). Rejected.
- 2026-10-09, GigaAM, decoder with NEON tiles (`std::arch::aarch64`; plain `dot` elsewhere): a
  tile computes 2 rows × 2 frames or 4 rows × 1 frame, each dot with `dot`'s 16 lanes and
  pairwise sum, so every value is bit-identical. The joint scores frames in pairs against one
  predictor output (the frame after a token is scored again, ~8% extra). Decoder (direct timer)
  ru 5.0 → 3.6, ru-short 11.7 (4 × 1 tiles alone) → 10.3, ru-long 38.5 → 27.2 ms; on ru-long
  4 × 1 tiles alone gave 31.3 ms and spans of 4 frames with 1 × 4 tiles 29.8 ms (18% rescored).
  Warm A/B/A: ru 45.7 / 44.1 / 45.8, ru-short 101.6 / 97.5 / 101.9 (another B 102.2, A 104.8),
  ru-long 315.2 / 306.2 / 315.8 ms. Fixtures exact. Kept. ru-long after: pair spans 13.9 ms
  (458), LSTM + predictor 9.8 ms (143 steps, 3.2 MB of f32 weights each, ~57 GB/s from L2),
  single-frame joints 3.2 ms. The reference spends 19 ms (Accelerate sgemm on 32-frame spans).
  The GGUF stores these weights as Q8_0 (expanded to f32 at load; not tried in-register).
- 2026-10-09, GigaAM, attention specialized for head_dim 48 (constant instead of the param, so
  the d and c loops are fixed): attention 49.8 vs 48.9 ms (GPU timestamps, ru-long). No gain;
  rejected.
- 2026-10-09, GigaAM, attention in 64-key blocks (scores in two 32-key halves, softmax with two
  keys per lane, one output load/store and one rescale check per 64 keys): attention 56.4 → 48.9
  ms; warm ru-long A/B/A 322.3 / 316.6 / 321.9 ms, ru-short B/A/B 102.0 / 102.5 / 102.3 ms.
  Kept.
- 2026-10-09, GigaAM, attention skips the output rescale when a row's running max didn't move
  (exact: the factor would be 1): attention 71.4 → 56.4 ms; warm ru-long A/B/A 337.1 / 322.1 /
  337.4 ms. Fixtures exact. Kept.
- 2026-10-09, GigaAM, Q8_0 GEMM cost (GPU timestamps, ru-long, timing ablations whose outputs
  were wrong). q·k 1536←768 Q8_0 takes 27.2 ms vs 16.7 for the F16 pointwise1 of the same
  shape, but: without the per-row scale lookup q·k 25.6 ms (FFN 72.2 / 70.7 → 69.5 / 69.0);
  without the int8 unpack 25.4 ms; with the q·k weights stored as F16 ~25 ms; reading `y`
  instead of the RoPE output 27.1 ms. So dequantizing Q8_0 costs ~8% (5% the scale lookup), and
  the remaining gap depends on where the GEMM runs in the block: attention-out (Q8_0) takes 9.0
  ms like pointwise2 (F16, 8.8 ms) while v (Q8_0, same shape as attention-out) takes 10–12.5 ms.
  Per-dispatch timestamps with one pass per dispatch likely charge part of the previous kernel's
  write-back to the next one. Nothing to keep; scales next to the data would add ~12 MB, which
  the ru-long footprint margin doesn't allow, for ≤ 5%. The load-time warm-up profile (M = 2)
  shows FFN down at 0.37 ms per dispatch: only 12 workgroups each walk 192 K steps.

- 2026-10-09, GigaAM, first call: warm-up at load, an encoder run on 8 silent frames. Phase 1
  build (A) vs this (B), back to back: first call ru 92.1 / 50.8 / 71.9 / 51.1 ms, ru-short
  146.6 / 110.0 / 131.2 / 108.5 ms, ru-long 369.9 / 348.1 / 369.4 / 348.9 ms; load 65–68 →
  77–81 ms. Kept. How it was found: with all weights flushed at load (temporary
  `queue.submit([])` + wait) the first call fell to 53 ms and load rose to 96 ms, so the cost
  was wgpu's deferred staging copies; with mapped weight buffers those are gone, but the first
  GPU use of pipelines and weights still costs ~41 ms of GPU time even on 8 frames, of which
  20–30 ms showed in the first call. Encode + submit were < 1 ms either way.
- 2026-10-09, GigaAM, first call: arena allocated for 30 s at load (warm-up on it) instead of on
  demand: ru 51.5 / 50.1 ms, ru-short 120.6 / 108.4 ms, no better than the on-demand arena and
  +26 MB on short clips. Rejected.
- 2026-10-09, GigaAM, peak memory: the GGUF header is read alone and every tensor is streamed
  from the file in 278 KB pieces straight into weight buffers created mapped with
  `MAPPABLE_PRIMARY_BUFFERS` (unified memory, no staging copy); Q8_0 scales stay f16 (exact,
  −12 MB); the arena drops its im2col and v buffers (`h`, `qk` and `y` take those roles, −12 MB
  on ru-long). Peak footprint 920 / 922 / 961 → 307 / 318 / 346 MB (reference 338 / 340 / 349
  MB); max RSS 312 / 317 / 326 MB. Warm unchanged: shared weight buffers vs private ones
  334.6 vs 334.0 / 335.6 ms on ru-long. Kept. Skipping the unused portable pipelines saved only
  0.5 MB: rejected.
- 2026-10-09, GigaAM, profile with GPU timestamps per dispatch kind (temporary tool, removed):
  see "Profile" above.

- 2026-10-09, GigaAM, phase 1 baseline recorded in "Current state" and "Targets" above. Peak
  footprint wavo 878 / 880 / 925 MiB vs reference 322 / 324 / 333 MiB (max RSS 609 / 602 / 620
  vs 356 / 357 / 366 MiB).
- 2026-10-09, GigaAM, decoder: removed the helper thread (a library must not busy-wait a core,
  and a panic in the main loop would hang `thread::scope` on it). Decoder 26 → 38.5 ms on
  ru-long; warm ru-long ~+12 ms. Single-threaded replacements, all slower than the plain matvec
  (38.7 ms, direct timer, A/B back to back):
  - 4 joint rows per pass over z, so each z chunk loads once for 4 rows (exact per row):
    generic `dots<const R>` 94 ms (accumulators spilled), index loops 43.5 ms (bounds checks per
    row, z reloaded per row), zipped row iterators 61.6 ms (LLVM's SLP vectorizer transposes
    across the 4 rows with shuffles). Rejected; needs NEON intrinsics to win.
  - Spans of frames sharing one predictor output, rows outer and frames inner (the reference's
    lookahead): the earlier 8-frame try was 20 ms slower. Why: the matvec is bound by L1 loads
    and FMAs, not by L2 traffic, so keeping a weight row in L1 across frames saves nothing unless
    the loaded registers are reused across frames (same codegen problem as above), while the
    frames after each emission are rescored (estimated ~20% more dots on ru-long: 143 tokens
    over 845 frames). Not retried.
- 2026-10-09, GigaAM, state after Task 2 (back to back, warm median / min, first call, load):
  wavo ru 43.7 / 43.5, 80.7, 75 ms; ru-short 97.6 / 97.3, 122.2, 69 ms; ru-long 317.9 / 316.6,
  346.1, 71 ms. Reference ru 43.7 / 43.3, 58.9, 123 ms; ru-short 99.6 / 98.6, 103.0, 124 ms;
  ru-long 310.4 / 308.3, 318.9, 128 ms. The first call is 25–37 ms over warm (arena allocation,
  bind groups, buffer zero-init); the reference's only 10–15 ms.
- 2026-10-09, GigaAM, decoder: a helper thread scores the upper half of the joint vocabulary,
  spinning on an atomic step counter for the duration of `decode`. Decoder 38 → 26 ms on
  ru-long (direct timer). Warm B/A/B: ru-long 330.5 / 343.1 / 334.5, ru-short A/B/A 107.9 /
  102.7 / 106.8 ms. Kept; bit-exact with one thread.
- 2026-10-09, GigaAM, GEMM F16 operands (a and dequantized W rounded to f16 in shared memory,
  f16×f16→f32 cooperative matrices, `SHADER_F16`): fixtures unchanged but slower, ru-long 353.1
  vs 329.3, ru 45.9 vs 45.4 ms. Rejected.
- 2026-10-09, GigaAM, GEMM K step 32 instead of 16 (one Q8_0 block per step, half the barriers):
  ru-long B/A/B 347.8 / 330.0 / 347.2 ms. Rejected, as in research.
- 2026-10-09, GigaAM, flash attention without workgroup barriers: each subgroup walks all keys
  on its own, loads k and v fragments straight from memory and syncs only with
  `subgroupBarrier` around its softmax. ru-long B/A/B 331.3 / 357.9 / 329.7 ms (attention ~111
  → ~85 ms). Kept. Reading rows of qk/v past t (masked) needs them finite: the arena's are.
- 2026-10-09, GigaAM, flash attention variants on the barrier version, ru-long: q tile staged
  once in shared memory (arrays sized for head_dim ≤ 48) 329.0 vs 330.4, no gain, rejected;
  q fragments in an array of cooperative matrices 394.2 vs 375.9 (spills), rejected; shared
  memory 32 → 26 KB 374.4 vs 376.5, no gain, rejected; k/v staging loops without division by
  head_dim 370.4 vs 377.5 / 375.9, kept until the barrier-free kernel removed staging; softmax
  with lane = key and subgroupMax/Add instead of 2 lanes per row (16-way bank conflicts) 368.9
  / 370.1 / 368.4, no change alone, kept in the barrier-free kernel. An ablation without the
  softmax/rescale phase saved 47 of 111 ms: the serial phase between barriers was the cost.
- 2026-10-09, GigaAM, mel filterbank with the vectorized dot: mel 25 → 8 ms on ru-long. Kept.
- 2026-10-09, GigaAM, decoder dot: 32 accumulators 43 → 48 ms (the 32 scalar adds of the final
  sum dominate a 320-long dot); 16 FMA accumulators summed pairwise 43 → 38.5 ms. Kept.
- 2026-10-09, GigaAM, decoder: joint scored for spans of 8 frames per weight pass while the
  predictor output is unchanged. ru-long A/B/A 397.5 / 417.2 / 397.9 ms. Rejected: rescoring
  the frames after each token outweighs the cache reuse.
- 2026-10-09, GigaAM, GEMM weight type as a pipeline-overridable constant (one pipeline per
  type) instead of a runtime switch: fast ru A/B/A 56.9 / 50.0 / 56.8, ru-long 431.4 / 396.2 /
  429.7 ms. Kept. Portable path neutral (119.2 / 116.9 / 120.0). The one-entry portable GEMM is
  slower than Task 1's three-entry one (ru 85 → 118 ms); cause not found, fallback only.
- 2026-10-09, GigaAM, load: Q8_0 repack by per-block memcpy instead of a byte iterator (encoder
  upload 181 → 77 ms), the 16 blocks repacked and uploaded from 16 threads (77 → 28 ms), GGUF
  read in 8 parallel chunks (36 → 20 ms). Load 233 → 60–90 ms. Kept. Measured with temporary
  timers, 2–3 runs each, machine load average 3–5.
- 2026-10-09, GigaAM, fast paths: cooperative-matrix GEMM (research `gemm_fast` layout: 32×64
  tile, K16, F32 8×8, Q8_0/F16 dequantized while staging W), flash attention (64 queries × 32
  keys per workgroup, online softmax, output accumulator in shared memory), one subgroup per row
  for LayerNorm(+RoPE) and conv_glu. Warm portable → fast, A/B/A: ru 118.0 / 56.4 / 118.4,
  ru-short 280.2 / 128.7 / 283.1, ru-long 1046 / 427 / 1008 ms. Kept. Profile after it
  (ru-long, by skipping dispatches): GEMM ~245 ms (~1.5 TFLOP/s), attention ~109 ms, other GPU
  ~8 ms; CPU mel 25 ms, decoder 43 ms.
- 2026-10-09, GigaAM, Task 1 baseline (portable kernels, one compute pass per call): warm 85.1 /
  203.1 / 808.4 ms, load 0.23 s. Reference warm 46.4 / 99.6 / 311.2 ms. Profile (ru-long):
  attention 370 ms (scores matrix in memory), GEMM 425 ms, LayerNorm/conv 55 ms.
