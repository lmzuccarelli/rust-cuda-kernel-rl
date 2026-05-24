use crate::MAP_LOOKUP;
use custom_logger as log;
use std::process::Command;

pub trait ShellExecuteInterface {
    fn run(cmd: &str, args: Vec<&str>) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct ShellExecute {}

impl ShellExecuteInterface for ShellExecute {
    fn run(cmd: &str, args: Vec<&str>) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing command {}", cmd);
        let output = Command::new(cmd)
            .args(args)
            .output()
            .expect("failed to spawn shell");

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(stdout)
    }
}

#[allow(unused)]
fn get_item(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let hm_guard = MAP_LOOKUP.lock().map_err(|_| "mutex lock failed")?;
    let value = match hm_guard.as_ref() {
        Some(res) => {
            let item_value = res.get(name);
            match item_value {
                Some(final_value) => final_value,
                None => {
                    return Err(Box::from(format!(
                        "[get_item] hashmap lookup {} not found",
                        name
                    )));
                }
            }
        }
        None => {
            return Err(Box::from("[get_item] error validating hashmap lookup"));
        }
    };
    Ok(value.to_string())
}
