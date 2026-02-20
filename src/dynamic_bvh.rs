//! Dynamic AABB Tree (Incremental BVH)
//!
//! A self-balancing binary tree of AABBs for efficient broadphase collision
//! detection with dynamic objects. Unlike `LinearBvh` (rebuild-only), this tree
//! supports O(log n) insert, remove, and update operations.
//!
//! # Features
//!
//! - **Incremental updates**: Insert/remove/move bodies without full rebuild
//! - **Fat AABBs**: Enlarged margins reduce re-insertions for moving bodies
//! - **Tree rotations**: AVL-style balancing for O(log n) query performance
//! - **Deterministic**: Fixed-point AABB comparisons, stable sort

use crate::collider::AABB;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Null node sentinel
pub const NULL_NODE: u32 = u32::MAX;

/// Default AABB fat margin (extends AABB by this amount in each direction)
const FAT_MARGIN: Fix128 = Fix128 {
    hi: 0,
    lo: 0x8000000000000000,
}; // 0.5

/// A node in the dynamic AABB tree
#[derive(Clone, Debug)]
pub struct DynamicNode {
    /// Fat AABB (enlarged for movement prediction)
    pub aabb: AABB,
    /// Parent node index (NULL_NODE if root)
    pub parent: u32,
    /// Left child (NULL_NODE if leaf)
    pub left: u32,
    /// Right child (NULL_NODE if leaf)
    pub right: u32,
    /// Height (0 for leaf, max(left.height, right.height) + 1)
    pub height: i32,
    /// User data (body index for leaves, unused for internal)
    pub user_data: u32,
    /// Whether this node is a leaf
    pub is_leaf: bool,
}

impl DynamicNode {
    fn new_leaf(aabb: AABB, user_data: u32) -> Self {
        Self {
            aabb,
            parent: NULL_NODE,
            left: NULL_NODE,
            right: NULL_NODE,
            height: 0,
            user_data,
            is_leaf: true,
        }
    }

    fn new_internal() -> Self {
        Self {
            aabb: AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO),
            parent: NULL_NODE,
            left: NULL_NODE,
            right: NULL_NODE,
            height: 0,
            user_data: NULL_NODE,
            is_leaf: false,
        }
    }
}

/// Dynamic AABB Tree for incremental broadphase
pub struct DynamicAabbTree {
    /// Node pool
    nodes: Vec<DynamicNode>,
    /// Free list (indices of unused nodes)
    free_list: Vec<u32>,
    /// Root node index
    root: u32,
    /// AABB fattening margin
    pub margin: Fix128,
}

impl DynamicAabbTree {
    /// Create a new empty tree
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            root: NULL_NODE,
            margin: FAT_MARGIN,
        }
    }

    /// Insert a new AABB, returns the proxy (node) ID
    pub fn insert(&mut self, aabb: AABB, user_data: u32) -> u32 {
        let fat_aabb = self.fatten(aabb);
        let node_id = self.alloc_node();

        self.nodes[node_id as usize] = DynamicNode::new_leaf(fat_aabb, user_data);

        self.insert_leaf(node_id);
        node_id
    }

    /// Remove a proxy by its ID
    pub fn remove(&mut self, proxy_id: u32) {
        if proxy_id as usize >= self.nodes.len() {
            return;
        }
        self.remove_leaf(proxy_id);
        self.free_node(proxy_id);
    }

    /// Update a proxy's AABB. Returns true if the tree was modified.
    ///
    /// Only re-inserts if the tight AABB has left the fat AABB.
    pub fn update(&mut self, proxy_id: u32, new_aabb: AABB) -> bool {
        if proxy_id as usize >= self.nodes.len() {
            return false;
        }

        let fat = &self.nodes[proxy_id as usize].aabb;
        // Check if new AABB is still within fat AABB
        if fat.min.x <= new_aabb.min.x
            && fat.min.y <= new_aabb.min.y
            && fat.min.z <= new_aabb.min.z
            && fat.max.x >= new_aabb.max.x
            && fat.max.y >= new_aabb.max.y
            && fat.max.z >= new_aabb.max.z
        {
            return false; // Still within fat bounds
        }

        self.remove_leaf(proxy_id);
        self.nodes[proxy_id as usize].aabb = self.fatten(new_aabb);
        self.insert_leaf(proxy_id);
        true
    }

    /// Get user data for a proxy
    #[inline]
    pub fn user_data(&self, proxy_id: u32) -> u32 {
        self.nodes[proxy_id as usize].user_data
    }

    /// Get the AABB for a proxy
    #[inline]
    pub fn get_aabb(&self, proxy_id: u32) -> AABB {
        self.nodes[proxy_id as usize].aabb
    }

    /// Query all proxies overlapping the given AABB
    pub fn query(&self, aabb: &AABB) -> Vec<u32> {
        let mut result = Vec::new();
        if self.root == NULL_NODE {
            return result;
        }

        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);

        while let Some(node_id) = stack.pop() {
            if node_id == NULL_NODE {
                continue;
            }

            let node = &self.nodes[node_id as usize];
            if !node.aabb.intersects(aabb) {
                continue;
            }

            if node.is_leaf {
                result.push(node.user_data);
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }

        result
    }

    /// Query with callback (avoids allocation)
    pub fn query_callback<F: FnMut(u32)>(&self, aabb: &AABB, mut callback: F) {
        if self.root == NULL_NODE {
            return;
        }

        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);

        while let Some(node_id) = stack.pop() {
            if node_id == NULL_NODE {
                continue;
            }

            let node = &self.nodes[node_id as usize];
            if !node.aabb.intersects(aabb) {
                continue;
            }

            if node.is_leaf {
                callback(node.user_data);
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
    }

    /// Find all potentially overlapping pairs
    pub fn find_pairs(&self) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();

        if self.root == NULL_NODE {
            return pairs;
        }

        // Collect all leaf nodes
        let mut leaves = Vec::new();
        self.collect_leaves(self.root, &mut leaves);

        // For each leaf, query the tree
        for &leaf_id in &leaves {
            let aabb = &self.nodes[leaf_id as usize].aabb;
            let ud_a = self.nodes[leaf_id as usize].user_data;

            self.query_callback(aabb, |ud_b| {
                if ud_a < ud_b {
                    pairs.push((ud_a, ud_b));
                }
            });
        }

        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Number of active proxies (leaf nodes)
    pub fn proxy_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.is_leaf && n.user_data != NULL_NODE)
            .count()
    }

    /// Total node count (including internal)
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.free_list.len()
    }

    /// Tree height
    pub fn height(&self) -> i32 {
        if self.root == NULL_NODE {
            0
        } else {
            self.nodes[self.root as usize].height
        }
    }

    // =========== Internal methods ===========

    fn fatten(&self, aabb: AABB) -> AABB {
        let m = Vec3Fix::new(self.margin, self.margin, self.margin);
        AABB::new(aabb.min - m, aabb.max + m)
    }

    fn alloc_node(&mut self) -> u32 {
        if let Some(id) = self.free_list.pop() {
            id
        } else {
            let id = self.nodes.len() as u32;
            self.nodes.push(DynamicNode::new_internal());
            id
        }
    }

    fn free_node(&mut self, node_id: u32) {
        self.nodes[node_id as usize].height = -1;
        self.nodes[node_id as usize].user_data = NULL_NODE;
        self.nodes[node_id as usize].is_leaf = false;
        self.nodes[node_id as usize].left = NULL_NODE;
        self.nodes[node_id as usize].right = NULL_NODE;
        self.nodes[node_id as usize].parent = NULL_NODE;
        self.free_list.push(node_id);
    }

    fn insert_leaf(&mut self, leaf: u32) {
        if self.root == NULL_NODE {
            self.root = leaf;
            self.nodes[leaf as usize].parent = NULL_NODE;
            return;
        }

        // Find best sibling using surface area heuristic
        let leaf_aabb = self.nodes[leaf as usize].aabb;
        let mut sibling = self.root;

        while !self.nodes[sibling as usize].is_leaf {
            let left = self.nodes[sibling as usize].left;
            let right = self.nodes[sibling as usize].right;

            let area = self.nodes[sibling as usize].aabb.surface_area();
            let combined = leaf_aabb.union(&self.nodes[sibling as usize].aabb);
            let combined_area = combined.surface_area();

            let cost = combined_area * Fix128::from_int(2);
            let inheritance_cost = (combined_area - area) * Fix128::from_int(2);

            let cost_left = self.child_insertion_cost(left, &leaf_aabb, inheritance_cost);
            let cost_right = self.child_insertion_cost(right, &leaf_aabb, inheritance_cost);

            if cost < cost_left && cost < cost_right {
                break;
            }

            sibling = if cost_left < cost_right { left } else { right };
        }

        // Create new parent
        let old_parent = self.nodes[sibling as usize].parent;
        let new_parent = self.alloc_node();
        self.nodes[new_parent as usize] = DynamicNode::new_internal();
        self.nodes[new_parent as usize].parent = old_parent;
        self.nodes[new_parent as usize].aabb = leaf_aabb.union(&self.nodes[sibling as usize].aabb);
        self.nodes[new_parent as usize].height = self.nodes[sibling as usize].height + 1;

        if old_parent != NULL_NODE {
            if self.nodes[old_parent as usize].left == sibling {
                self.nodes[old_parent as usize].left = new_parent;
            } else {
                self.nodes[old_parent as usize].right = new_parent;
            }
        } else {
            self.root = new_parent;
        }

        self.nodes[new_parent as usize].left = sibling;
        self.nodes[new_parent as usize].right = leaf;
        self.nodes[sibling as usize].parent = new_parent;
        self.nodes[leaf as usize].parent = new_parent;

        // Walk up and fix heights + AABBs + balance
        self.fix_upwards(new_parent);
    }

    fn child_insertion_cost(&self, child: u32, leaf_aabb: &AABB, inheritance: Fix128) -> Fix128 {
        let combined = leaf_aabb.union(&self.nodes[child as usize].aabb);
        if self.nodes[child as usize].is_leaf {
            combined.surface_area() + inheritance
        } else {
            let old_area = self.nodes[child as usize].aabb.surface_area();
            let new_area = combined.surface_area();
            (new_area - old_area) + inheritance
        }
    }

    fn remove_leaf(&mut self, leaf: u32) {
        if leaf == self.root {
            self.root = NULL_NODE;
            return;
        }

        let parent = self.nodes[leaf as usize].parent;
        let grand_parent = self.nodes[parent as usize].parent;
        let sibling = if self.nodes[parent as usize].left == leaf {
            self.nodes[parent as usize].right
        } else {
            self.nodes[parent as usize].left
        };

        if grand_parent != NULL_NODE {
            // Reconnect sibling to grandparent
            if self.nodes[grand_parent as usize].left == parent {
                self.nodes[grand_parent as usize].left = sibling;
            } else {
                self.nodes[grand_parent as usize].right = sibling;
            }
            self.nodes[sibling as usize].parent = grand_parent;
            self.free_node(parent);

            self.fix_upwards(grand_parent);
        } else {
            self.root = sibling;
            self.nodes[sibling as usize].parent = NULL_NODE;
            self.free_node(parent);
        }
    }

    fn fix_upwards(&mut self, start: u32) {
        let mut node_id = start;
        while node_id != NULL_NODE {
            node_id = self.balance(node_id);

            let left = self.nodes[node_id as usize].left;
            let right = self.nodes[node_id as usize].right;

            if left != NULL_NODE && right != NULL_NODE {
                let lh = self.nodes[left as usize].height;
                let rh = self.nodes[right as usize].height;
                self.nodes[node_id as usize].height = 1 + if lh > rh { lh } else { rh };
                self.nodes[node_id as usize].aabb = self.nodes[left as usize]
                    .aabb
                    .union(&self.nodes[right as usize].aabb);
            }

            node_id = self.nodes[node_id as usize].parent;
        }
    }

    /// AVL-style tree rotation for balancing
    fn balance(&mut self, node_id: u32) -> u32 {
        if self.nodes[node_id as usize].is_leaf || self.nodes[node_id as usize].height < 2 {
            return node_id;
        }

        let left = self.nodes[node_id as usize].left;
        let right = self.nodes[node_id as usize].right;

        let balance_factor = self.nodes[right as usize].height - self.nodes[left as usize].height;

        if balance_factor > 1 {
            self.rotate_left(node_id)
        } else if balance_factor < -1 {
            self.rotate_right(node_id)
        } else {
            node_id
        }
    }

    fn rotate_left(&mut self, node_id: u32) -> u32 {
        let right = self.nodes[node_id as usize].right;
        let right_left = self.nodes[right as usize].left;
        let right_right = self.nodes[right as usize].right;
        let parent = self.nodes[node_id as usize].parent;

        // Right becomes new parent
        self.nodes[right as usize].left = node_id;
        self.nodes[right as usize].parent = parent;
        self.nodes[node_id as usize].parent = right;

        if parent != NULL_NODE {
            if self.nodes[parent as usize].left == node_id {
                self.nodes[parent as usize].left = right;
            } else {
                self.nodes[parent as usize].right = right;
            }
        } else {
            self.root = right;
        }

        // Rotate based on children heights
        let rl_h = if right_left != NULL_NODE {
            self.nodes[right_left as usize].height
        } else {
            -1
        };
        let rr_h = if right_right != NULL_NODE {
            self.nodes[right_right as usize].height
        } else {
            -1
        };

        if rl_h > rr_h {
            self.nodes[right as usize].right = right_left;
            self.nodes[node_id as usize].right = right_right;
            if right_right != NULL_NODE {
                self.nodes[right_right as usize].parent = node_id;
            }
            if right_left != NULL_NODE {
                self.nodes[right_left as usize].parent = right;
            }
        } else {
            self.nodes[node_id as usize].right = right_left;
            if right_left != NULL_NODE {
                self.nodes[right_left as usize].parent = node_id;
            }
        }

        // Update AABBs and heights
        let left = self.nodes[node_id as usize].left;
        let nr = self.nodes[node_id as usize].right;
        if left != NULL_NODE && nr != NULL_NODE {
            self.nodes[node_id as usize].aabb = self.nodes[left as usize]
                .aabb
                .union(&self.nodes[nr as usize].aabb);
            let lh = self.nodes[left as usize].height;
            let rh = self.nodes[nr as usize].height;
            self.nodes[node_id as usize].height = 1 + if lh > rh { lh } else { rh };
        }

        let rl = self.nodes[right as usize].left;
        let rr = self.nodes[right as usize].right;
        if rl != NULL_NODE && rr != NULL_NODE {
            self.nodes[right as usize].aabb = self.nodes[rl as usize]
                .aabb
                .union(&self.nodes[rr as usize].aabb);
            let lh = self.nodes[rl as usize].height;
            let rh = self.nodes[rr as usize].height;
            self.nodes[right as usize].height = 1 + if lh > rh { lh } else { rh };
        }

        right
    }

    fn rotate_right(&mut self, node_id: u32) -> u32 {
        let left = self.nodes[node_id as usize].left;
        let left_left = self.nodes[left as usize].left;
        let left_right = self.nodes[left as usize].right;
        let parent = self.nodes[node_id as usize].parent;

        self.nodes[left as usize].right = node_id;
        self.nodes[left as usize].parent = parent;
        self.nodes[node_id as usize].parent = left;

        if parent != NULL_NODE {
            if self.nodes[parent as usize].left == node_id {
                self.nodes[parent as usize].left = left;
            } else {
                self.nodes[parent as usize].right = left;
            }
        } else {
            self.root = left;
        }

        let ll_h = if left_left != NULL_NODE {
            self.nodes[left_left as usize].height
        } else {
            -1
        };
        let lr_h = if left_right != NULL_NODE {
            self.nodes[left_right as usize].height
        } else {
            -1
        };

        if lr_h > ll_h {
            self.nodes[left as usize].left = left_right;
            self.nodes[node_id as usize].left = left_left;
            if left_left != NULL_NODE {
                self.nodes[left_left as usize].parent = node_id;
            }
            if left_right != NULL_NODE {
                self.nodes[left_right as usize].parent = left;
            }
        } else {
            self.nodes[node_id as usize].left = left_right;
            if left_right != NULL_NODE {
                self.nodes[left_right as usize].parent = node_id;
            }
        }

        // Update AABBs and heights
        let nl = self.nodes[node_id as usize].left;
        let right = self.nodes[node_id as usize].right;
        if nl != NULL_NODE && right != NULL_NODE {
            self.nodes[node_id as usize].aabb = self.nodes[nl as usize]
                .aabb
                .union(&self.nodes[right as usize].aabb);
            let lh = self.nodes[nl as usize].height;
            let rh = self.nodes[right as usize].height;
            self.nodes[node_id as usize].height = 1 + if lh > rh { lh } else { rh };
        }

        let ll = self.nodes[left as usize].left;
        let lr = self.nodes[left as usize].right;
        if ll != NULL_NODE && lr != NULL_NODE {
            self.nodes[left as usize].aabb = self.nodes[ll as usize]
                .aabb
                .union(&self.nodes[lr as usize].aabb);
            let lh = self.nodes[ll as usize].height;
            let rh = self.nodes[lr as usize].height;
            self.nodes[left as usize].height = 1 + if lh > rh { lh } else { rh };
        }

        left
    }

    fn collect_leaves(&self, node_id: u32, leaves: &mut Vec<u32>) {
        if node_id == NULL_NODE {
            return;
        }

        if self.nodes[node_id as usize].is_leaf {
            leaves.push(node_id);
        } else {
            self.collect_leaves(self.nodes[node_id as usize].left, leaves);
            self.collect_leaves(self.nodes[node_id as usize].right, leaves);
        }
    }
}

impl Default for DynamicAabbTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_aabb(x: i64, y: i64, z: i64) -> AABB {
        AABB::new(
            Vec3Fix::from_int(x, y, z),
            Vec3Fix::from_int(x + 1, y + 1, z + 1),
        )
    }

    #[test]
    fn test_insert_and_query() {
        let mut tree = DynamicAabbTree::new();

        let _p0 = tree.insert(make_aabb(0, 0, 0), 0);
        let _p1 = tree.insert(make_aabb(10, 10, 10), 1);
        let _p2 = tree.insert(make_aabb(20, 20, 20), 2);

        assert_eq!(tree.proxy_count(), 3);

        // Query near first body
        let results = tree.query(&make_aabb(-1, -1, -1));
        assert!(results.contains(&0));
        assert!(!results.contains(&2));

        // Query large area
        let all = tree.query(&AABB::new(
            Vec3Fix::from_int(-100, -100, -100),
            Vec3Fix::from_int(100, 100, 100),
        ));
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_remove() {
        let mut tree = DynamicAabbTree::new();

        let _p0 = tree.insert(make_aabb(0, 0, 0), 0);
        let p1 = tree.insert(make_aabb(5, 5, 5), 1);
        let _p2 = tree.insert(make_aabb(10, 10, 10), 2);

        assert_eq!(tree.proxy_count(), 3);

        tree.remove(p1);
        assert_eq!(tree.proxy_count(), 2);

        let all = tree.query(&AABB::new(
            Vec3Fix::from_int(-100, -100, -100),
            Vec3Fix::from_int(100, 100, 100),
        ));
        assert!(!all.contains(&1));
    }

    #[test]
    fn test_update_no_reinsert() {
        let mut tree = DynamicAabbTree::new();

        let p0 = tree.insert(make_aabb(0, 0, 0), 0);

        // Small movement within fat AABB margin — should NOT reinsert
        let tiny_move = AABB::new(
            Vec3Fix::new(Fix128::from_ratio(1, 10), Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(
                Fix128::ONE + Fix128::from_ratio(1, 10),
                Fix128::ONE,
                Fix128::ONE,
            ),
        );
        let reinserted = tree.update(p0, tiny_move);
        assert!(!reinserted, "Small move should not trigger reinsert");
    }

    #[test]
    fn test_update_reinsert() {
        let mut tree = DynamicAabbTree::new();

        let p0 = tree.insert(make_aabb(0, 0, 0), 0);

        // Large movement outside fat AABB — should reinsert
        let far_move = make_aabb(100, 100, 100);
        let reinserted = tree.update(p0, far_move);
        assert!(reinserted, "Large move should trigger reinsert");

        // Should still be queryable at new position
        let results = tree.query(&make_aabb(99, 99, 99));
        assert!(results.contains(&0));
    }

    #[test]
    fn test_find_pairs() {
        let mut tree = DynamicAabbTree::new();

        // Two overlapping bodies
        tree.insert(
            AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(2, 2, 2)),
            0,
        );
        tree.insert(
            AABB::new(Vec3Fix::from_int(1, 1, 1), Vec3Fix::from_int(3, 3, 3)),
            1,
        );
        // One far away
        tree.insert(make_aabb(100, 100, 100), 2);

        let pairs = tree.find_pairs();
        assert!(
            pairs.contains(&(0, 1)),
            "Overlapping bodies should form a pair"
        );
        assert!(
            !pairs.contains(&(0, 2)),
            "Far body should not pair with body 0"
        );
    }

    #[test]
    fn test_tree_balance() {
        let mut tree = DynamicAabbTree::new();

        // Insert many bodies — tree should remain balanced
        for i in 0..100 {
            tree.insert(make_aabb(i * 3, 0, 0), i as u32);
        }

        assert_eq!(tree.proxy_count(), 100);
        // Height should be O(log n) — for 100 nodes, ~7-10
        assert!(
            tree.height() < 20,
            "Tree should be balanced, height={}",
            tree.height()
        );
    }

    #[test]
    fn test_empty_tree() {
        let tree = DynamicAabbTree::new();
        assert_eq!(tree.proxy_count(), 0);
        assert_eq!(tree.height(), 0);
        assert!(tree.query(&make_aabb(0, 0, 0)).is_empty());
        assert!(tree.find_pairs().is_empty());
    }
}
