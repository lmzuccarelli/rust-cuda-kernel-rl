use crate::cli::schema::{Cli, Commands};
use crate::config::load::Parameters;
use crate::config::load::{ConfigInterface, ImplConfigInterface};
use crate::routers::*;
use crate::utils::shell::{ShellExecute, ShellExecuteInterface};
use crate::workflow::controller::{Controller, ControllerInterface};
use clap::Parser;
use custom_logger as log;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use mimalloc::MiMalloc;
use std::collections::HashMap;
use std::env;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Mutex;
use tokio::net::TcpListener;

mod cli;
mod config;
mod inference;
mod kernel;
mod routers;
mod utils;
mod workflow;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// used for lookup in read mode only
static MAP_LOOKUP: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

// This application must be executed on a server with GPU capabilities
// The premise here is to be able use this service as remote
// enabling the compilation, testing and profiling of a cuda kernel
// the results can then be used for reinforcment learning
fn main() {
    let args = Cli::parse();

    // read and parse config
    let impl_config = ImplConfigInterface {};
    let res_params = impl_config.read(args.config);
    let parameters = match res_params {
        Ok(params) => params,
        Err(e) => {
            eprintln!("[main] config file error: {}", e);
            std::process::exit(1);
        }
    };

    let level = match parameters.log_level.as_str() {
        "debug" => log::LevelFilter::Debug,
        "trace" => log::LevelFilter::Trace,
        &_ => log::LevelFilter::Info,
    };

    // setup logging
    if let Err(e) = log::Logging::new().with_level(level).init() {
        // log is broken so use eprintln!
        // no use continuing
        eprintln!("[main] error {}", e);
        std::process::exit(1);
    }

    let (sub_command, parameters) = match &args.command {
        Some(Commands::Llm {}) => ("llm".to_string(), parameters.clone()),
        Some(Commands::Gpu {}) => {
            // check for cuda in PATH and LD_LIBRARY_PATH envars
            match env::var("PATH") {
                Ok(val) => {
                    let contents = val.to_string();
                    if contents.is_empty() || !contents.contains("cuda") {
                        log::error!("[main] no cuda binaries found in path");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    log::error!("[main] PATH envar not set {}", e);
                    std::process::exit(1);
                }
            }
            match env::var("LD_LIBRARY_PATH") {
                Ok(val) => {
                    let contents = val.to_string();
                    if contents.is_empty()
                        || !contents.contains("cuda")
                        || !contents.contains("CUPTI")
                    {
                        log::error!("[main] no cuda libraries found in LD_LIBRARY_PATH");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    log::error!("[main] LD_LIBRARY_PATH envar not set {}", e);
                    std::process::exit(1);
                }
            }
            // parameters used in service
            let mut hm: HashMap<String, String> = HashMap::new();

            // get gpu info
            // this determines the number of concurrent kernels to evaluate
            let impl_shell_res = ShellExecute::run("nvidia-smi", vec!["--list-gpus"]);
            match impl_shell_res {
                Ok(gpus) => {
                    // preserve output
                    println!("{}", gpus);
                    hm.insert("gpus".to_string(), gpus);
                }
                Err(e) => {
                    log::error!(
                        "[main] could not resolve nvidia-smi (is the nvidia driver installed) {}",
                        e
                    );
                    std::process::exit(1);
                }
            }

            let gpu_arch_res = ShellExecute::run("nvidia-smi", vec!["--query-gpu=compute_cap"]);
            // preserve output
            println!("{}", gpu_arch_res.unwrap_or("no arch".to_string()));

            *MAP_LOOKUP.lock().unwrap() = Some(hm.clone());
            ("gpu".to_string(), parameters)
        }
        Some(Commands::Compile {}) => {
            let impl_shell_res = ShellExecute::run("nvcc", vec!["--version"]);
            match impl_shell_res {
                Ok(res) => {
                    log::debug!("nvcc version {}", res)
                }
                Err(e) => {
                    log::error!("[main] please ensure nvcc is installed correctly {}", e);
                    std::process::exit(1);
                }
            }
            ("compile".to_string(), parameters)
        }
        Some(Commands::Controller {}) => ("controller".to_string(), parameters),

        None => {
            log::error!(
                "please ensure you use the correct sub-command (available sub-commands are one of gpu,compile,llm,controller)"
            );
            std::process::exit(1);
        }
    };

    let result = execute(sub_command.clone(), parameters);
    match result {
        Ok(_) => log::info!("[main] terminating process {}", sub_command),
        Err(err) => {
            log::error!("{}", err);
            std::process::exit(1);
        }
    }
}

#[tokio::main]
pub async fn execute(
    command: String,
    parameters: Parameters,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("application : {}", env!("CARGO_PKG_NAME"));
    log::info!("author      : {}", env!("CARGO_PKG_AUTHORS"));
    log::info!("version     : {}", env!("CARGO_PKG_VERSION"));
    log::info!("server      : {}", command);

    // check if we are in workflow controller mode or launch service mode
    if command == "controller" {
        // check health endpoints
        Controller::get_health(parameters.clone()).await?;
        // endpoints are good lets get the cuda-kernel baseline
        Controller::get_baseline(parameters).await?;
        Ok(())
    } else {
        let port = get_port(command.clone(), parameters)?;
        log::info!("port        : {}", port);
        let addr = SocketAddr::new(Ipv4Addr::new(0, 0, 0, 0).into(), port);
        log::info!("[execute] starting to serve on http://{}", addr);
        let listener = TcpListener::bind(addr).await?;
        match command.as_str() {
            "compile" => loop {
                let (stream, _) = listener.accept().await?;
                let io = TokioIo::new(stream);
                tokio::task::spawn(async move {
                    if let Err(err) = http1::Builder::new()
                        .serve_connection(io, service_fn(compiler::endpoints))
                        .await
                    {
                        log::error!("[execute] error serving connection: {:?}", err);
                    }
                });
            },
            "gpu" => loop {
                let (stream, _) = listener.accept().await?;
                let io = TokioIo::new(stream);
                tokio::task::spawn(async move {
                    if let Err(err) = http1::Builder::new()
                        .serve_connection(io, service_fn(gpu::endpoints))
                        .await
                    {
                        log::error!("[execute] error serving connection: {:?}", err);
                    }
                });
            },
            "llm" => loop {
                let (stream, _) = listener.accept().await?;
                let io = TokioIo::new(stream);
                tokio::task::spawn(async move {
                    if let Err(err) = http1::Builder::new()
                        .serve_connection(io, service_fn(llm::endpoints))
                        .await
                    {
                        log::error!("[execute] error serving connection: {:?}", err);
                    }
                });
            },
            &_ => {
                log::error!("[execute] invalid service");
                std::process::exit(1);
            }
        }
    }
}

// utility

fn get_port(command: String, parameters: Parameters) -> Result<u16, Box<dyn std::error::Error>> {
    let port = match command.as_str() {
        "compile" => parameters
            .compile_server_url
            .split(":")
            .nth(2)
            .unwrap_or("3201")
            .parse::<u16>()?,
        "gpu" => parameters
            .gpu_server_url
            .split(":")
            .nth(2)
            .unwrap_or("3202")
            .parse::<u16>()?,
        "llm" => parameters
            .llm_server_url
            .split(":")
            .nth(2)
            .unwrap_or("3200")
            .parse::<u16>()?,
        &_ => {
            return Err(Box::from(
                "invalid command (valid commands are compile,gpu,llm)",
            ));
        }
    };
    Ok(port)
}
