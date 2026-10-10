// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! A status line on stderr while `wavo run` works: a spinner, the current step and the elapsed
//! seconds. Only on a terminal, so pipes and files get nothing new.

use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

pub struct Progress(Mutex<String>);

impl Progress {
  pub fn set(&self, step: impl Into<String>) {
    *self.0.lock().unwrap() = step.into();
  }
}

/// Runs `work` while a thread redraws the line every 100 ms, and erases the line before
/// returning, so whatever is printed next starts on a clean line.
pub fn show<T>(work: impl FnOnce(&Progress) -> T) -> T {
  let progress = Progress(Mutex::new(String::new()));
  if !io::stderr().is_terminal() {
    return work(&progress);
  }
  let (start, (stop, stopped)) = (Instant::now(), mpsc::channel::<()>());
  std::thread::scope(|s| {
    let progress = &progress;
    s.spawn(move || {
      let mut drawn = false;
      for spin in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏".chars().cycle() {
        if stopped.recv_timeout(Duration::from_millis(100)) != Err(RecvTimeoutError::Timeout) {
          break;
        }
        let step = progress.0.lock().unwrap().clone();
        let line = format!("\r{spin} {step} {:.1}s\x1b[K", start.elapsed().as_secs_f64());
        drawn |= io::stderr().write_all(line.as_bytes()).is_ok();
      }
      if drawn {
        let _ = io::stderr().write_all(b"\r\x1b[2K");
      }
    });
    let out = work(progress);
    drop(stop);
    out
  })
}
