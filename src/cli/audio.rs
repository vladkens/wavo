// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Audio files to the 16 kHz mono samples the models take.

use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;

const RATE: usize = 16000;

/// A format or codec wavo doesn't decode, such as Opus or WebM.
#[derive(Debug)]
pub struct Unsupported;

impl std::fmt::Display for Unsupported {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    f.write_str("not a wav, mp3, m4a, flac or ogg file")
  }
}

impl std::error::Error for Unsupported {}

/// Decodes wav, mp3, m4a/aac, flac or ogg/vorbis, averages the channels and resamples to 16 kHz.
/// A 16 kHz mono file passes through untouched, so its samples are exactly the WAV's.
pub fn read(path: &str) -> Result<Vec<f32>> {
  let file = File::open(path).with_context(|| format!("reading {path}"))?;
  let ext = Path::new(path).extension().and_then(|e| e.to_str());
  pcm(Box::new(file), ext, None).map_err(|e| match e.is::<Unsupported>() {
    true => anyhow!("{path}: {e}"),
    false => e.context(format!("reading {path}")),
  })
}

/// [`read`] for the bytes of a file whose name ends in `ext`, a hint at the format, of up to
/// `max_secs` seconds: a small compressed file can decode to more samples than fit in memory.
/// Fails with [`Unsupported`] for a format it doesn't decode.
pub fn decode_bytes(bytes: Vec<u8>, ext: Option<&str>, max_secs: usize) -> Result<Vec<f32>> {
  pcm(Box::new(Cursor::new(bytes)), ext, Some(max_secs))
}

fn pcm(source: Box<dyn MediaSource>, ext: Option<&str>, max: Option<usize>) -> Result<Vec<f32>> {
  let (pcm, rate) = decode(source, ext, max).map_err(|e| match e.downcast_ref() {
    Some(Error::Unsupported(_)) => Unsupported.into(),
    _ => e,
  })?;
  if rate == RATE {
    return Ok(pcm);
  }

  let mut resampler = Fft::<f32>::new(rate, RATE, 1024, 1, FixedSync::Input)?;
  let input = InterleavedSlice::new(&pcm, 1, pcm.len())?;
  Ok(resampler.process_all(&input, pcm.len(), None)?.take_data())
}

/// Mono samples and their rate, of up to `max_secs` seconds.
fn decode(
  source: Box<dyn MediaSource>,
  ext: Option<&str>,
  max_secs: Option<usize>,
) -> Result<(Vec<f32>, usize)> {
  let mut hint = Hint::new();
  if let Some(ext) = ext {
    hint.with_extension(ext);
  }
  let source = MediaSourceStream::new(source, Default::default());
  let options = (FormatOptions::default(), MetadataOptions::default());
  let mut format = symphonia::default::get_probe().probe(&hint, source, options.0, options.1)?;
  let track = format.default_track(TrackType::Audio).context("no audio track")?;
  let params = track.codec_params.as_ref().and_then(|p| p.audio()).context("not audio")?.clone();
  let (id, rate) = (track.id, params.sample_rate.context("unknown sample rate")? as usize);
  let codecs = symphonia::default::get_codecs();
  let mut decoder = codecs.make_audio_decoder(&params, &AudioDecoderOptions::default())?;
  let (mut pcm, mut frame) = (Vec::new(), Vec::new());
  while let Some(packet) = format.next_packet()? {
    if packet.track_id != id {
      continue;
    }
    let audio = match decoder.decode(&packet) {
      Ok(audio) => audio,
      Err(Error::DecodeError(_)) => continue, // a corrupt packet: skip it, as players do
      Err(e) => return Err(e.into()),
    };
    let channels = audio.spec().channels().count();
    audio.copy_to_vec_interleaved(&mut frame);
    match channels {
      1 => pcm.extend_from_slice(&frame),
      n => pcm.extend(frame.chunks(n).map(|f| f.iter().sum::<f32>() / n as f32)),
    }

    if let Some(max) = max_secs.filter(|&max| pcm.len() > max * rate) {
      bail!("the audio is longer than {:.1} hours: send it in parts", max as f64 / 3600.0);
    }
  }
  Ok((pcm, rate))
}

#[cfg(test)]
mod tests {
  use std::f32::consts::TAU;

  use super::*;

  /// Writes `samples` as a WAV in the temp dir, decodes it with `read` and deletes it.
  fn decode_wav<S: hound::Sample + Copy>(
    name: &str,
    spec: hound::WavSpec,
    samples: &[S],
  ) -> Vec<f32> {
    let path = std::env::temp_dir().join(format!("wavo-{}-{name}.wav", std::process::id()));
    let mut wav = hound::WavWriter::create(&path, spec).unwrap();
    samples.iter().for_each(|&s| wav.write_sample(s).unwrap());
    wav.finalize().unwrap();
    let pcm = read(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();
    pcm
  }

  fn spec(channels: u16, sample_rate: u32, format: hound::SampleFormat) -> hound::WavSpec {
    let bits_per_sample = if format == hound::SampleFormat::Int { 16 } else { 32 };
    hound::WavSpec { channels, sample_rate, bits_per_sample, sample_format: format }
  }

  /// The fixtures' samples: a 16 kHz mono PCM16 WAV read with hound as `s as f32 / 32768.0`.
  #[test]
  fn pcm16_16k_mono_is_bit_exact() {
    let mut x = 1u32;
    let mut samples = vec![i16::MIN, i16::MIN + 1, -1, 0, 1, i16::MAX - 1, i16::MAX];
    samples.extend((0..48000).map(|_| {
      x = x.wrapping_mul(1664525).wrapping_add(1013904223);
      (x >> 16) as i16
    }));
    let pcm = decode_wav("pcm16", spec(1, 16000, hound::SampleFormat::Int), &samples);
    let expected: Vec<u32> = samples.iter().map(|&s| (s as f32 / 32768.0).to_bits()).collect();
    assert!(pcm.iter().map(|s| s.to_bits()).eq(expected), "samples differ from hound's");
  }

  #[test]
  fn bytes_decode_as_files_do() {
    let mut wav = Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(&mut wav, spec(1, 16000, hound::SampleFormat::Int)).unwrap();
    [0i16, 1000, -1000].iter().for_each(|&s| w.write_sample(s).unwrap());
    w.finalize().unwrap();
    let pcm = decode_bytes(wav.into_inner(), None, 1).unwrap();
    assert_eq!(pcm, [0.0, 1000.0 / 32768.0, -1000.0 / 32768.0]);
    // WebM's magic, then nothing a decoder knows
    let e = decode_bytes(b"\x1a\x45\xdf\xa3 then junk".repeat(100), Some("webm"), 1).unwrap_err();
    assert!(e.is::<Unsupported>(), "{e:#}");
    let mut wav = Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(&mut wav, spec(2, 8000, hound::SampleFormat::Int)).unwrap();
    (0..2 * 8001).for_each(|_| w.write_sample(0i16).unwrap());
    w.finalize().unwrap();
    let e = decode_bytes(wav.into_inner(), None, 1).unwrap_err().to_string();
    assert_eq!(e, "the audio is longer than 0.0 hours: send it in parts", "1 s and a sample");
  }

  #[test]
  fn stereo_44k_is_averaged_and_resampled() {
    let tone = |t: f32| (TAU * 440.0 * t).sin();
    let samples: Vec<f32> =
      (0..44100).map(|i| tone(i as f32 / 44100.0)).flat_map(|s| [s, 0.5 * s]).collect();
    let pcm = decode_wav("stereo", spec(2, 44100, hound::SampleFormat::Float), &samples);
    assert!(pcm.len().abs_diff(16000) <= 1, "{} samples", pcm.len());
    // The resampler leaves a delay of a fraction of a sample (~11 µs here).
    for (i, s) in pcm.iter().enumerate().take(15000).skip(1000) {
      let expected = 0.75 * tone(i as f32 / 16000.0);
      assert!((s - expected).abs() < 0.03, "sample {i}: {s} vs {expected}");
    }
    let peak = pcm[1000..15000].iter().fold(0f32, |m, s| m.max(s.abs()));
    assert!((peak - 0.75).abs() < 0.005, "peak {peak}");
  }
}
