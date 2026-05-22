use crate::config::load::WorkItem;
use custom_logger as log;
use orx_parallel::*;
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

        let results: Vec<(usize, bool, String)> = (0..1usize)
            .par()
            .map(|i| {
                println!("creating job {i}");
                let output = Command::new("bash")
                    .args(vec![
                        "-c",
                        "execute.sh",
                        &work_item.name,
                        &work_item.gpu_arch,
                    ])
                    .output()
                    .expect("failed to spawn shell");

                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                (i, output.status.success(), stdout)
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
