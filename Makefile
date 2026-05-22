# for static build using x86_64-unknown-linux-gnu ensure glibc-static (rhel based) is installed

.PHONY: all clean-all build build-local verify fmt

all: clean-all build

build-static:
	RUSTFLAGS="-C target-feature=+crt-static -Clink-arg=-fuse-ld=lld" cargo build --target x86_64-unknown-linux-gnu --release --verbose 

build:
	cargo build --release 

verify:
	cargo clippy --all-targets --all-features

fmt:
	rustfmt --check src/*.rs --edition 2024

clean-all:
	rm -rf cargo-test*
	cargo clean
	rm -rf ./target/debug
