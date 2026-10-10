// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Audio files to the 16 kHz mono samples the models take.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

const RATE: usize = 16000;

/// Decodes wav, mp3, m4a/aac, flac or ogg/vorbis, averages the channels and resamples to 16 kHz.
/// A 16 kHz mono file passes through untouched, so its samples are exactly the WAV's.
pub fn read(path: &str) -> Result<Vec<f32>> {
  let (pcm, rate) = decode(path).map_err(|e| match e.downcast_ref() {
    Some(Error::Unsupported(_)) => anyhow!("{path}: not a wav, mp3, m4a, flac or ogg file"),
    _ => e.context(format!("reading {path}")),
  })?;
  if rate == RATE {
    return Ok(pcm);
  }
  let mut resampler = Fft::<f32>::new(rate, RATE, 1024, 1, FixedSync::Input)?;
  let input = InterleavedSlice::new(&pcm, 1, pcm.len())?;
  Ok(resampler.process_all(&input, pcm.len(), None)?.take_data())
}

/// Mono samples and their rate.
fn decode(path: &str) -> Result<(Vec<f32>, usize)> {
  let mut hint = Hint::new();
  if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
    hint.with_extension(ext);
  }
  let source = MediaSourceStream::new(Box::new(File::open(path)?), Default::default());
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
  }
  Ok((pcm, rate))
}

#[cfg(test)]
mod tests {
  use std::f32::consts::TAU;

  use super::*;

  #[test]
  fn fixture_wavs_are_bit_exact() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("3rd/transcribe.cpp/samples");
    for sample in ["jfk", "dots", "jobs-silence", "ru", "ru-short", "ru-long", "uk-short"] {
      let path = root.join(format!("{sample}.wav"));
      let mut wav = hound::WavReader::open(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
      let expected: Vec<u32> =
        wav.samples::<i16>().map(|s| (s.unwrap() as f32 / 32768.0).to_bits()).collect();
      let got: Vec<u32> =
        read(path.to_str().unwrap()).unwrap().iter().map(|s| s.to_bits()).collect();
      assert!(got == expected, "{sample}: samples differ from hound's");
    }
  }

  #[test]
  fn stereo_44k_is_averaged_and_resampled() {
    let path = std::env::temp_dir().join(format!("wavo-{}-stereo.wav", std::process::id()));
    let spec = hound::WavSpec {
      channels: 2,
      sample_rate: 44100,
      bits_per_sample: 32,
      sample_format: hound::SampleFormat::Float,
    };
    let mut wav = hound::WavWriter::create(&path, spec).unwrap();
    let tone = |t: f32| (TAU * 440.0 * t).sin();
    for i in 0..44100 {
      let s = tone(i as f32 / 44100.0);
      wav.write_sample(s).unwrap();
      wav.write_sample(0.5 * s).unwrap();
    }
    wav.finalize().unwrap();
    let pcm = read(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();
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
