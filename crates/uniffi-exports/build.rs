fn main() {
    uniffi::generate_scaffolding("./src/lib.rs").expect("UniFFI scaffolding generation");
}
