use crate::MAP_LOOKUP;
use crate::inference::llm::{LlmClaude, LlmInterface};
use crate::inference::llm::{LlmInterfaceOpenApi, LlmOpenApi};
use custom_logger as log;
use http::{Method, Request, Response, StatusCode};
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};

pub async fn endpoints(req: Request<Incoming>) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let mut response = Response::new(Full::default());
    let request = req.uri().path();
    log::debug!("{}", request);
    match *req.method() {
        Method::POST => match request {
            x if x.contains("/v1/prompt/claude") => {
                let data = req.into_body().collect().await?.to_bytes();
                let prompt_res = String::from_utf8(data.to_vec());
                match prompt_res {
                    Ok(prompt) => {
                        let result = LlmClaude::run(prompt).await;
                        match result {
                            Ok(content) => {
                                *response.body_mut() = Full::from(content);
                            }
                            Err(err) => {
                                log::error!("[endpoints] llm prompt request error {}", err);
                                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                                *response.body_mut() = Full::from(format!(
                                    "[endpoints] llm prompt request error {}\n",
                                    err
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("[endpoints] prompt request parsing body error {}", err);
                        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        *response.body_mut() = Full::from(format!(
                            "[endpoints] prompt request parsing body error {}\n",
                            err
                        ));
                    }
                }
            }
            x if x.contains("/v1/prompt/openapi") => {
                let data = req.into_body().collect().await?.to_bytes();
                let prompt_res = String::from_utf8(data.to_vec());
                match prompt_res {
                    Ok(prompt) => {
                        let model = get_item("model").unwrap_or("none".to_string());
                        let url = get_item("url").unwrap_or("".to_string());
                        let token = get_item("token").unwrap_or("".to_string());
                        let result = LlmOpenApi::run(prompt, url, token, model).await;
                        match result {
                            Ok(content) => {
                                *response.body_mut() = Full::from(content);
                            }
                            Err(err) => {
                                log::error!("[endpoints] llm prompt request error {}", err);
                                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                                *response.body_mut() = Full::from(format!(
                                    "[endpoints] llm prompt request error {}\n",
                                    err
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("[endpoints] prompt request parsing body error {}", err);
                        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        *response.body_mut() = Full::from(format!(
                            "[endpoints] prompt request parsing body error {}\n",
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
                    r##"{{ "status": "ok", "appplication": "{}", "service": "llm-inference", "version": "{}" }}"##,
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
