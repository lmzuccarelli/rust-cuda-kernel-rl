use crate::config::load::WorkItem;
use custom_logger as log;
use std::env;
use std::fs;
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

        env::set_current_dir(format!("out/{}/build", work_item.name))?;
        let ld_lib = env::var("LD_LIBRARY_PATH")?;

        let output = Command::new("sudo")
            .arg(format!("LD_LIBRARY_PATH={}", ld_lib))
            .arg("ncu")
            .arg("--set")
            .arg("full")
            .arg("-o")
            .arg("profile")
            .arg("-f")
            .arg("main")
            .output()
            .expect("failed to execute ncu profile agent");

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let elapsed = start.elapsed();
        log::info!("completed task in {:?}", elapsed);

        if !output.status.success() {
            return Err(Box::from(stderr.to_string()));
        }

        // preserve output
        println!("{}", stdout);

        let output = Command::new("sudo")
            .arg(format!("LD_LIBRARY_PATH={}", ld_lib))
            .arg("ncu")
            .arg("--import")
            .arg("profile.ncu-rep")
            .arg("--page")
            .arg("details")
            .output()
            .expect("failed to execute ncu convert profile agent");

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let elapsed = start.elapsed();
        log::info!("completed task in {:?}", elapsed);

        if !output.status.success() {
            return Err(Box::from(stderr.to_string()));
        }

        // preserve output
        // println!("{}", stdout);
        // write the final report to disk
        fs::write("output.profile", stdout)?;

        Ok("exit => 0".to_string())
    }
}
