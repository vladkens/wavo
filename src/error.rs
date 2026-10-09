// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
#[derive(Debug, thiserror::Error)]
pub enum Error {
  #[error(transparent)]
  Io(#[from] std::io::Error),
  #[error("invalid model: {0}")]
  Model(String),
  #[error("no GPU adapter: {0}")]
  Adapter(#[from] wgpu::RequestAdapterError),
  #[error("GPU device: {0}")]
  Device(#[from] wgpu::RequestDeviceError),
  #[error("GPU readback: {0}")]
  Readback(#[from] wgpu::BufferAsyncError),
  #[error("GPU readback: {0}")]
  Map(#[from] wgpu::MapRangeError),
  #[error("GPU poll: {0}")]
  Poll(#[from] wgpu::PollError),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Shorthand for `Error::Model` with a formatted message.
macro_rules! bail {
  ($($arg:tt)*) => {
    return Err($crate::error::Error::Model(format!($($arg)*)))
  };
}
pub(crate) use bail;
