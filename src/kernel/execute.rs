use crate::config::load::WorkItem;
use custom_logger as log;
use std::process::Command;
use std::time::Instant;

pub trait ExecuteInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Execute {}

impl ExecuteInterface for Execute {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing cuda-kernel binary");
        let start = Instant::now();

        let output = Command::new(format!("{}/build/main", work_item.target_dir)).output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let elapsed = start.elapsed();
        log::info!("[run] execute : completed task in {:?}", elapsed);

        if !output.status.success() {
            return Err(Box::from(stderr.to_string()));
        }

        // preserve output
        println!("{}", stdout);

        Ok(stdout)
    }
}
