# 🦀 A "Vector DB" in Rust

This is not intended to be used in production, nor is it very efficient. It's a naive implementation of a vector database using a k-d tree, but it works? There's also no rebalancing added, so performance will be degraded after many additions over time.

One day, it would be fun to use the C bindings for Google's ScaNN or Spotify's Annoy instead of this.

## Usage

Initialize the tree with a handful of vectors:

```rust
let data = vec![
    vec![1.0, 1.0],
    vec![2.0, 2.0],
    vec![3.0, 3.0],
    vec![4.0, 4.0],
    vec![5.0, 5.0],
]

let ids = vec!["a", "b", "c", "d", "e"]
    .into_iter()
    .map(|s| s.to_string())
    .collect::<Vec<String>>();

let tree = Arc::new(RwLock::new(KDTree::new(&data, &ids, 0).unwrap()));
```

Add a vector:

```rust
KDTree::add(&tree, "f", &[6.0, 6.0]).await;
```

Nearest neighbor:

```rust
let target = &[5.5, 5.5];
let (nearest_id, nearest_point) = KDTree::nearest_neighbor(&tree, target).await;
```
