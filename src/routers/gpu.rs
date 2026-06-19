use crate::kernel::execute::{Execute, ExecuteInterface};
use crate::kernel::profile::{Profile, ProfileInterface};
use custom_logger as log;
use http::{Method, Request, Response, StatusCode};
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};

pub async fn endpoints(req: Request<Incoming>) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let mut response = Response::new(Full::default());
    let request = req.uri().path();
    log::debug!("[endpoint] gpu request {}", request);
    match *req.method() {
        Method::POST => match request {
            x if x.contains("/v1/execute") => {
                let data = req.into_body().collect().await?.to_bytes();
                let work_item_res = serde_json::from_slice(&data);
                match work_item_res {
                    Ok(work_item) => {
                        let workflow_res = Execute::run(work_item).await;
                        match workflow_res {
                            Ok(content) => {
                                *response.body_mut() = Full::from(content);
                            }
                            Err(err) => {
                                log::error!("[endpoints] executing cuda kernel {}", err);
                                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                                *response.body_mut() = Full::from(format!(
                                    "[endpoints] executing cuda kernel error {}\n",
                                    err
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("[endpoints] executing cuda kernel parsing body {}", err);
                        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        *response.body_mut() = Full::from(format!(
                            "[endpoints] executing cuda kernel parsing body error {}\n",
                            err
                        ));
                    }
                }
            }
            x if x.contains("/v1/profile") => {
                let data = req.into_body().collect().await?.to_bytes();
                let work_item_res = serde_json::from_slice(&data);
                match work_item_res {
                    Ok(work_item) => {
                        let workflow_res = Profile::run(work_item).await;
                        match workflow_res {
                            Ok(content) => {
                                *response.body_mut() = Full::from(content);
                            }
                            Err(err) => {
                                log::error!("[endpoints] profiling cuda kernel {}", err);
                                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                                *response.body_mut() = Full::from(format!(
                                    "[endpoints] profiling cuda kernel error {}\n",
                                    err
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("[endpoints] profiling cuda kernel parsing body {}", err);
                        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        *response.body_mut() = Full::from(format!(
                            "[endpoints] profiling cuda kernel parsing body error {}\n",
                            err
                        ));
                    }
                }
            }
            &_ => {}
        },
        Method::GET => match request {
            x if x.contains("/v1/health") => {
                let mut content = format!(
                    r##"{{ "status": "ok", "appplication": "{}", "service": "gpu", "version": "{}" }}"##,
                    env!("CARGO_PKG_NAME"),
                    env!("CARGO_PKG_VERSION"),
                );
                content.push('\n');
                *response.body_mut() = Full::from(content);
            }
            &_ => {}
        },
        _ => {
            log::error!("[endpoints] method/endpoint not implemented");
            *response.body_mut() = Full::from("[endpoints] method/endpoint not implmented\n");
            *response.status_mut() = StatusCode::NOT_FOUND;
        }
    };
    Ok(response)
}
