// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Core ML through its public Objective-C API: compile an `.mlpackage`, load the compiled model on
//! the CPU and Neural Engine, predict on F16 arrays.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use half::f16;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{AnyThread, ClassType};
use objc2_core_ml::{
  MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
  MLModelConfiguration, MLMultiArray, MLMultiArrayDataType, MLNeuralEngineComputeDevice,
};
use objc2_foundation::{
  NSArray, NSDictionary, NSError, NSNumber, NSObjectProtocol, NSProcessInfo, NSString, NSURL,
};

use crate::{Error, Result};

fn error(what: &str, e: Retained<NSError>) -> Error {
  Error::NeuralEngine(format!("{what}: {}", e.localizedDescription()))
}

fn url(path: &Path) -> Retained<NSURL> {
  NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

/// Whether this Mac has a Neural Engine Core ML can use, on macOS 15 or later (the opset the
/// programs use). Virtual machines have none.
pub fn available() -> bool {
  autoreleasepool(|_| {
    if NSProcessInfo::processInfo().operatingSystemVersion().majorVersion < 15 {
      return false;
    }
    let class = MLNeuralEngineComputeDevice::class();
    // A class method, so a binary built for older macOS still starts there (no link-time symbol).
    let devices = unsafe { MLModel::availableComputeDevices() };
    devices.iter().any(|d| d.isKindOfClass(class))
  })
}

/// The OS version with its build, such as `Version 15.8.1 (Build 24H32)`.
pub fn os_version() -> String {
  autoreleasepool(|_| NSProcessInfo::processInfo().operatingSystemVersionString().to_string())
}

/// The chip's name, such as `Apple M2`; empty when the system doesn't say.
pub fn chip() -> String {
  unsafe extern "C" {
    fn sysctlbyname(
      name: *const std::ffi::c_char,
      old: *mut std::ffi::c_void,
      len: *mut usize,
      new: *const std::ffi::c_void,
      new_len: usize,
    ) -> std::ffi::c_int;
  }

  let (mut buf, mut len) = ([0u8; 128], 128);
  let name = c"machdep.cpu.brand_string";
  // SAFETY: the buffer holds `len` bytes, and sysctl writes at most that many.
  let ok =
    unsafe { sysctlbyname(name.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null(), 0) };
  let text = if ok == 0 { &buf[..len] } else { &[] };
  String::from_utf8_lossy(text).trim_end_matches('\0').trim().to_string()
}

/// Compiles an `.mlpackage` and returns the compiled `.mlmodelc` folder, which Core ML leaves in
/// a temporary folder.
pub fn compile(package: &Path) -> Result<PathBuf> {
  autoreleasepool(|_| {
    #[allow(deprecated)] // the replacement takes a completion block
    let out = unsafe { MLModel::compileModelAtURL_error(&url(package)) };
    let out = out.map_err(|e| error("compiling the Core ML model", e))?;
    let path =
      out.path().ok_or_else(|| Error::NeuralEngine("compiled model has no path".into()))?;
    Ok(PathBuf::from(path.to_string()))
  })
}

/// A compiled model loaded on the CPU and Neural Engine.
pub struct Model {
  model: Retained<MLModel>,
  /// One prediction at a time, as the GPU path's arena.
  lock: Mutex<()>,
}

// SAFETY: MLModel is immutable once loaded, and predictions take turns through `lock`.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
  /// Loads a compiled `.mlmodelc`. The first load of a model compiles it for the Neural Engine
  /// (seconds), which the system caches for later loads of the same path.
  pub fn load(path: &Path) -> Result<Self> {
    autoreleasepool(|_| {
      let config = unsafe { MLModelConfiguration::new() };
      unsafe { config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
      let model =
        unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url(path), &config) };
      let model = model.map_err(|e| error("loading the Core ML model", e))?;
      Ok(Self { model, lock: Mutex::new(()) })
    })
  }

  /// Runs the model on F16 inputs `(name, shape, values)` and returns the output `name`, a
  /// row-major `[rows][cols]` array, as f32.
  pub fn predict(&self, inputs: &[(&str, &[usize], &[f32])], output: &str) -> Result<Vec<f32>> {
    let _turn = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
    autoreleasepool(|_| {
      let mut keys = Vec::new();
      let mut values: Vec<Retained<AnyObject>> = Vec::new();
      for &(name, shape, data) in inputs {
        keys.push(NSString::from_str(name));
        let array = array(shape, data)?;
        let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
        values.push(Retained::into_super(Retained::into_super(value)));
      }
      let keys: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
      let dict = NSDictionary::from_retained_objects(&keys, &values);
      let features = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
          MLDictionaryFeatureProvider::alloc(),
          &dict,
        )
      };
      let features = features.map_err(|e| error("Core ML inputs", e))?;
      let features = ProtocolObject::<dyn MLFeatureProvider>::from_ref(&*features);
      let out = unsafe { self.model.predictionFromFeatures_error(features) };
      let out = out.map_err(|e| error("Core ML prediction", e))?;
      let value = unsafe { out.featureValueForName(&NSString::from_str(output)) };
      let array = value.and_then(|v| unsafe { v.multiArrayValue() });
      let array = array.ok_or_else(|| Error::NeuralEngine(format!("no output {output}")))?;
      Ok(read(&array))
    })
  }
}

fn numbers(v: &[usize]) -> Retained<NSArray<NSNumber>> {
  let v: Vec<Retained<NSNumber>> = v.iter().map(|&d| NSNumber::new_isize(d as isize)).collect();
  NSArray::from_retained_slice(&v)
}

/// A contiguous F16 array of `data`.
fn array(shape: &[usize], data: &[f32]) -> Result<Retained<MLMultiArray>> {
  let array = unsafe {
    MLMultiArray::initWithShape_dataType_error(
      MLMultiArray::alloc(),
      &numbers(shape),
      MLMultiArrayDataType::Float16,
    )
  };
  let array = array.map_err(|e| error("Core ML input array", e))?;
  #[allow(deprecated)] // the replacements take blocks
  let ptr = unsafe { array.dataPointer() }.cast::<f16>();
  let out = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), data.len()) };
  out.iter_mut().zip(data).for_each(|(y, &x)| *y = f16::from_f32(x));
  Ok(array)
}

/// A 2-D F16 or F32 output, which may have padded rows, as contiguous f32 rows.
fn read(array: &MLMultiArray) -> Vec<f32> {
  let dims = |v: Retained<NSArray<NSNumber>>| v.iter().map(|n| n.as_isize() as usize).collect();
  let (shape, strides): (Vec<usize>, Vec<usize>) =
    unsafe { (dims(array.shape()), dims(array.strides())) };
  let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
  let row_stride = strides[strides.len() - 2];
  #[allow(deprecated)]
  let ptr = unsafe { array.dataPointer() };
  let len = (rows - 1) * row_stride + cols;
  let mut out = Vec::with_capacity(rows * cols);
  if unsafe { array.dataType() } == MLMultiArrayDataType::Float32 {
    let data = unsafe { std::slice::from_raw_parts(ptr.cast::<f32>().as_ptr(), len) };
    (0..rows).for_each(|r| out.extend_from_slice(&data[r * row_stride..][..cols]));
  } else {
    let data = unsafe { std::slice::from_raw_parts(ptr.cast::<f16>().as_ptr(), len) };
    (0..rows).for_each(|r| out.extend(data[r * row_stride..][..cols].iter().map(|x| x.to_f32())));
  }
  out
}
