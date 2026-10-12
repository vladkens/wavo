// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Core ML outside macOS: no Neural Engine, and no model can be loaded.

use std::path::{Path, PathBuf};

use crate::{Error, Result};

pub fn available() -> bool {
  false
}

pub fn os_version() -> String {
  String::new()
}

pub fn compile(_: &Path) -> Result<PathBuf> {
  Err(Error::NeuralEngine("Core ML is only on macOS".into()))
}

/// Never constructed.
pub enum Model {}

impl Model {
  pub fn load(_: &Path) -> Result<Self> {
    Err(Error::NeuralEngine("Core ML is only on macOS".into()))
  }

  pub fn predict(&self, _: &[(&str, &[usize], &[f32])], _: &str) -> Result<Vec<f32>> {
    match *self {}
  }
}
