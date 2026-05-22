use crate::config::load::WorkItem;
use custom_logger as log;
use orx_parallel::*;
use std::process::Command;
use std::time::Instant;

pub trait ProfileInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Profile {}

impl ProfileInterface for Profile {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] profiling cuda-kernel");
        let start = Instant::now();

        let results: Vec<(usize, bool, String)> = (0..1usize)
            .par()
            .map(|i| {
                let output = Command::new("bash")
                    .args(vec![
                        "-c",
                        "profile.sh",
                        &work_item.name,
                        &work_item.gpu_arch,
                    ])
                    .output()
                    .expect("failed to spawn shell");

                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                (i, output.status.success(), stderr)
            })
            .collect();

        let elapsed = start.elapsed();
        log::info!("completed task in {:?}", elapsed);

        for (_i, ok, out) in &results {
            if !*ok {
                return Err(Box::from(out.to_string()));
            }
        }

        Ok("exit => 0".to_string())
    }
}
