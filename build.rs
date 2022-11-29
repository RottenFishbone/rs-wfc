use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=resources/kernel.cu");
    match Command::new("nvcc")
        .args(&["-ptx", "resources/kernel.cu", "-o", "resources/kernel.ptx",
                "-arch=sm_60"])
        .status() {
            Ok(msg) => {
                if !msg.success() {
                    panic!()
                }
            },
            Err(_) => panic!(),
        }
}
