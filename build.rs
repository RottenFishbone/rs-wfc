use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=resources/kernel.cu");
    Command::new("nvcc")
        .args(&["-ptx", "resources/kernel.cu", "-o", "resources/kernel.ptx"])
        .status().unwrap();
}
