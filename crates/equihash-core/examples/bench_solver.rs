use equihash_core::challenge::build_input;
use equihash_core::solver::{try_nonce, BaseState};
use std::time::Instant;

fn main() {
    let challenge = [1u8; 32];
    let miner = [2u8; 32];
    let height = 0u64;
    let input = build_input(&challenge, &miner, height);
    let base = BaseState::new(96, 5, &input).unwrap();

    // Warmup
    let nonce = [0u8; 32];
    let _ = try_nonce(&base, &input, &nonce);

    let n_attempts = 50;
    let mut total_solutions = 0usize;
    let start = Instant::now();
    for i in 0..n_attempts {
        let mut nonce = [0u8; 32];
        nonce[..8].copy_from_slice(&(i as u64).to_le_bytes());
        if try_nonce(&base, &input, &nonce).is_some() {
            total_solutions += 1;
        }
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_secs_f64() / n_attempts as f64;
    println!("attempts: {}  solutions: {}  elapsed: {:.2}s  per: {:.1}ms  rate: {:.1} nps",
        n_attempts, total_solutions, elapsed.as_secs_f64(), per * 1000.0, 1.0 / per);
}
