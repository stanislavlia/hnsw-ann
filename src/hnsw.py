import numpy as np
from typing import List, Callable
import heapq

def cosine_distance(v1, v2):
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - cosine_sim

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

class Node():
    def __init__(self, vector, id):
        self.vector = vector
        self.id = id
        self.neighbors = set()

    def add_neighbor(self, node_id):
        self.neighbors.add(node_id)

    def remove_neighbor(self, node_id):
        self.neighbors.remove(node_id)

    def copy(self):
        """Return a new Node with same id and vector but empty neighbors."""
        return Node(vector=self.vector, id=self.id)

    def __repr__(self):
        return f"Node(id={self.id}, neighbors={self.neighbors})"

class NSW():
    def __init__(self, M: int = 16, ef_construction: int = 200, distance_func: Callable = cosine_distance):
        self.M = M
        self.Mmax = 2 * M
        self.ef_construction = ef_construction
        self.distance_func = distance_func
        self.nodes: dict[int, Node] = {}
        self.entry_point: int | None = None

    def _search(self, query: np.ndarray, entry_id: int, ef: int):
        entry_dist = self.distance_func(query, self.nodes[entry_id].vector)
        candidates = []
        heapq.heappush(candidates, (entry_dist, entry_id))
        found = []
        heapq.heappush(found, (-entry_dist, entry_id))
        visited = {entry_id}

        while candidates:
            current_dist, current_id = heapq.heappop(candidates)
            current_node = self.nodes[current_id]
            worst_dist_found = -found[0][0]
            if current_dist > worst_dist_found:
                break
            for n_id in current_node.neighbors:
                if n_id in visited:
                    continue
                neigh_node = self.nodes[n_id]
                n_dist = self.distance_func(query, neigh_node.vector)
                visited.add(n_id)
                worst_dist_found = -found[0][0]
                if n_dist < worst_dist_found or len(found) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(found, (-n_dist, n_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted([(-neg_dist, nid) for neg_dist, nid in found])

    def _select_neighbors(self, candidates: List[tuple], M: int) -> List[Node]:
        candidates = sorted(candidates)
        if not candidates:
            return []
        selected_ids = [candidates[0][1]]
        for dist_to_q, node_id in candidates[1:]:
            if len(selected_ids) >= M:
                break
            candidate_node = self.nodes[node_id]
            accept = True
            for sid in selected_ids:
                if self.distance_func(candidate_node.vector, self.nodes[sid].vector) < dist_to_q:
                    accept = False
                    break
            if accept:
                selected_ids.append(node_id)
        return [self.nodes[nid] for nid in selected_ids]

    def insert(self, node: Node):
        # Insert a COPY so each layer has independent neighbor sets
        local_node = node.copy()
        self.nodes[local_node.id] = local_node

        if self.entry_point is None:
            self.entry_point = local_node.id
            return

        candidates = self._search(query=local_node.vector, entry_id=self.entry_point, ef=self.ef_construction)
        neighbors: List[Node] = self._select_neighbors(candidates, M=self.M)

        for neighbor in neighbors:
            local_node.add_neighbor(neighbor.id)
            neighbor.add_neighbor(local_node.id)
            if len(neighbor.neighbors) > self.Mmax:
                cands = [(self.distance_func(neighbor.vector, self.nodes[nid].vector), nid)
                         for nid in neighbor.neighbors]
                pruned = self._select_neighbors(cands, M=self.M)
                neighbor.neighbors = set(n.id for n in pruned)

    def search(self, query, k, ef=50):
        if self.entry_point is None:
            return []
        return self._search(query, self.entry_point, max(ef, k))[:k]


class HNSW():
    def __init__(self, M, ef_construction, distance_func):
        self.M = M
        self.layers: List[NSW] = []
        self.ef_construction = ef_construction
        self.distance_func = distance_func
        self.entry_point_id: int | None = None
        self.mL = 1 / np.log(M)
        self.top_level = 0
        self.current_id = 0

    def get_random_level(self) -> int:
        return int(-np.log(np.random.random()) * self.mL)

    def insert(self, vector):
        id = self.current_id
        self.current_id += 1
        node = Node(vector=vector, id=id)
        level = self.get_random_level()

        # ensure enough layers exist
        while len(self.layers) <= level:
            self.layers.append(
                NSW(M=self.M, ef_construction=self.ef_construction, distance_func=self.distance_func)
            )

        # first node — insert into all its layers and set entry point
        if self.entry_point_id is None:
            for i in range(level + 1):
                self.layers[i].insert(node=node)
            self.entry_point_id = id
            self.top_level = level
            return

        # update global entry point if new top level reached
        if level > self.top_level:
            self.top_level = level
            self.entry_point_id = node.id

        entry_id = self.entry_point_id

        # phase 1 — layers above node's level: greedy search only, no insert
        for layer_idx in range(self.top_level, level, -1):
            if entry_id in self.layers[layer_idx].nodes:
                results = self.layers[layer_idx]._search(query=vector, entry_id=entry_id, ef=1)
                entry_id = results[0][1]

        # phase 2 — at and below node's level: insert and connect
        for layer_idx in range(level, -1, -1):
            if entry_id in self.layers[layer_idx].nodes:
                self.layers[layer_idx].entry_point = entry_id
            self.layers[layer_idx].insert(node=node)
            # search from newly inserted node to get best entry for next layer
            results = self.layers[layer_idx]._search(query=vector, entry_id=node.id, ef=1)
            entry_id = results[0][1]

    def search(self, vector, k, ef=100):
        entry_id = self.entry_point_id
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            if layer_idx > 0:
                if entry_id in self.layers[layer_idx].nodes:
                    results = self.layers[layer_idx]._search(query=vector, entry_id=entry_id, ef=1)
                    entry_id = results[0][1]
            else:
                results = self.layers[layer_idx]._search(query=vector, entry_id=entry_id, ef=ef)[:k]
        return [self.layers[0].nodes[nid] for dist, nid in results]