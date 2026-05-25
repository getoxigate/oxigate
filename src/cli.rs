// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Shared CLI utilities used by both the community and Pro binaries.

use std::path::PathBuf;

use miette::{Result, miette};

/// Validates that `path` exists, is a regular file, and is readable.
///
/// Called after clap has parsed the `--config` value (including the default).
pub fn validate_config_path(path: PathBuf) -> Result<PathBuf> {
    if path.exists() && !path.is_file() {
        return Err(miette!("{} is a directory, not a file", path.display()));
    }
    std::fs::File::open(&path).map_err(|e| {
        miette!(
            "cannot open config file {}: {}; \
             pass --config <path> to specify a different location",
            path.display(),
            e
        )
    })?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn valid_file_accepted() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("oxigate.yaml");
        fs::write(&file, "").unwrap();
        assert_eq!(validate_config_path(file.clone()).unwrap(), file);
    }

    #[test]
    fn directory_rejected() {
        let dir = TempDir::new().unwrap();
        let err = validate_config_path(dir.path().to_path_buf()).unwrap_err();
        assert!(err.to_string().contains("directory"), "{err}");
    }

    #[test]
    fn missing_file_error_includes_hint() {
        let err = validate_config_path(PathBuf::from("/nonexistent/path/config.yaml")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cannot open config file"), "{msg}");
        assert!(msg.contains("--config"), "{msg}");
    }
}
