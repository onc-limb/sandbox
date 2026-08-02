use std::fs;
use std::collections::HashMap;

fn main() {
    let text = fs::read_to_string("data/sample.txt").expect("failed to read file");
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let normalized = word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string();
        *counts.entry(normalized).or_insert(0) += 1;
    }
    let mut sorted = counts.into_iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    for (word, count) in sorted.iter().take(10) {
        println!("{}: {}", word, count);
    }
}
