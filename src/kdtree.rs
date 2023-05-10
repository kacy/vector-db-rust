use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;
use tokio::sync::RwLock;

type SharedTree = Arc<RwLock<Box<KDTree>>>;

#[derive(Serialize, Deserialize, Debug)]
pub struct KDTree {
    id: String,
    point: Vec<f32>,
    left: Option<Box<KDTree>>,
    right: Option<Box<KDTree>>,
    dimension: usize,
}

impl KDTree {
    pub fn new(data: &[Vec<f32>], ids: &[String], depth: usize) -> Option<Box<KDTree>> {
        let k = data.get(0)?.len();
        let axis = depth % k;

        let mut data = data.to_vec();
        data.sort_by(|a, b| a[axis].partial_cmp(&b[axis]).unwrap());

        let median = data.len() / 2;

        let node = KDTree {
            id: ids[median].clone(),
            point: data[median].clone(),
            left: KDTree::new(&data[..median], &ids[..median], depth + 1),
            right: KDTree::new(&data[median + 1..], &ids[median + 1..], depth + 1),
            dimension: axis,
        };

        Some(Box::new(node))
    }

    fn nn_search(&self, target: &[f32]) -> (&String, &Vec<f32>, f32) {
        let axis = self.dimension;
        let current_point = &self.point;
        let current_distance = Self::distance_squared(current_point, target);

        let next_branch = if target[axis] < current_point[axis] {
            &self.left
        } else {
            &self.right
        };

        let mut best = if let Some(next) = next_branch {
            next.nn_search(target)
        } else {
            (&self.id, current_point, current_distance)
        };

        if current_distance < best.2 {
            best = (&self.id, current_point, current_distance);
        }

        let other_branch = if target[axis] < current_point[axis] {
            &self.right
        } else {
            &self.left
        };

        if other_branch.is_some() && target[axis].powf(2.0) < best.2 {
            let other_best = other_branch.as_ref().unwrap().nn_search(target);
            if other_best.1 < best.1 {
                best = other_best;
            }
        }

        best
    }

    fn distance_squared(p1: &[f32], p2: &[f32]) -> f32 {
        p1.iter()
            .zip(p2.iter())
            .map(|(x1, x2)| (x1 - x2).powf(2.0))
            .sum()
    }

    pub async fn save_to_file(tree: &SharedTree, filename: &str) -> std::io::Result<()> {
        let tree = tree.read().await;
        let mut file = File::create(filename)?;
        let serialized_tree = serde_json::to_string(&*tree)?;
        file.write_all(serialized_tree.as_bytes())?;
        Ok(())
    }

    pub async fn load_from_file(filename: &str) -> std::io::Result<SharedTree> {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let deserialized_tree: KDTree = serde_json::from_str(&contents)?;
        Ok(Arc::new(RwLock::new(Box::new(deserialized_tree))))
    }

    pub async fn nearest_neighbor(tree: &SharedTree, target: &[f32]) -> (String, Vec<f32>) {
        let tree = tree.read().await;
        let (id, point, _) = tree.nn_search(target);
        (id.clone(), point.clone())
    }

    pub async fn add(tree: &SharedTree, id: &str, point: &[f32]) {
        let mut tree = tree.write().await;
        tree.add_recursive(id, point, 0);
    }

    fn add_recursive(&mut self, id: &str, point: &[f32], depth: usize) {
        let axis = self.dimension;
        let direction = if point[axis] < self.point[axis] {
            &mut self.left
        } else {
            &mut self.right
        };

        if let Some(subtree) = direction {
            subtree.add_recursive(id, point, depth + 1)
        } else {
            *direction = KDTree::new(&[point.to_vec()], &[id.to_string()], depth + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
        ]
    }

    #[test]
    fn test_new() {
        let data = create_test_data();
        let ids = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let tree = KDTree::new(&data, &ids, 0).unwrap();
        assert_eq!(tree.point, vec![3.0, 3.0]);
        assert_eq!(tree.left.as_ref().unwrap().point, vec![2.0, 2.0]);
        assert_eq!(tree.right.as_ref().unwrap().point, vec![5.0, 5.0]);
    }

    #[test]
    fn test_nn_search() {
        let data = create_test_data();
        let ids = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let tree = KDTree::new(&data, &ids, 0).unwrap();
        let target = &[4.0, 4.0];
        let (_, nearest_neighbor, distance) = tree.nn_search(target);
        assert_eq!(nearest_neighbor, &vec![4.0, 4.0]);
        assert_eq!(distance, 0.0);
        let target = &[4.2, 4.2];
        let (_, nearest_neighbor, distance) = tree.nn_search(target);
        assert_eq!(nearest_neighbor, &vec![4.0, 4.0]);
        assert!(distance < 0.9999999);
    }

    #[test]
    fn test_distance_squared() {
        let p1 = &[1.0, 1.0];
        let p2 = &[4.0, 4.0];
        let distance = KDTree::distance_squared(p1, p2);
        assert_eq!(distance, 18.0);
    }

    #[tokio::test]
    async fn test_save_load_tree() {
        let data = create_test_data();
        let ids = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let kdtree = KDTree::new(&data, &ids, 0).unwrap();
        let tree = Arc::new(RwLock::new(kdtree));
        let filename = "test_tree.json";

        KDTree::save_to_file(&tree, filename).await.unwrap();
        let loaded_tree = KDTree::load_from_file(filename).await.unwrap();

        let target = &[4.0, 4.0];
        let nearest_neighbor = KDTree::nearest_neighbor(&loaded_tree, target).await;
        assert_eq!(nearest_neighbor.1, vec![4.0, 4.0]);
    }

    #[tokio::test]
    async fn test_add() {
        let data = create_test_data();
        let ids = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let tree = Arc::new(RwLock::new(KDTree::new(&data, &ids, 0).unwrap()));

        KDTree::add(&tree, "f", &[6.0, 6.0]).await;

        let target = &[5.5, 5.5];
        let (nearest_id, nearest_point) = KDTree::nearest_neighbor(&tree, target).await;
        assert_eq!(nearest_id, "f");
        assert_eq!(nearest_point, vec![6.0, 6.0]);
    }
}
