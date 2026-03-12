import numpy as np
from typing import List, Set, Callable
import heapq


def cosine_distance(v1, v2):
    
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - cosine_sim


class Node():
    def __init__(self,
                 vector,
                 id):
        
        self.vector = vector
        self.id = id
        self.neighbors = set()
    
    def add_neighbor(self, node):
        self.neighbors.add(node)

    def remove_neighbor(self, node):
        self.neighbors.remove(node)

    def __repr__(self):
        return f"Node(id={self.id}, neighbors={self.neighbors})"




class NSW():
    def __init__(self,
                M: int = 16,
                ef_construction: int = 200,
                distance_func: Callable = cosine_distance):
        
        self.M = M                          # max neighbors per node
        self.Mmax = 2 * M                   # max neighbors allowed before pruning
        self.ef_construction = ef_construction
        self.distance_func = distance_func

        self.nodes: dict[int, Node] = {}
        self.entry_point: int | None = None


    def _search(self, query: np.ndarray, entry_id: int, ef: int):
        """
        Beam search from entry_id toward query vector.
        Tracks ef closest nodes found.
        Returns list of (distance, node_id) sorted closest-first.
        """

        entry_dist = self.distance_func(query, self.nodes[entry_id].vector)
        
        
        #Min-Heap: nodes to explore with smallest distance
        candidates = []
        heapq.heappush(candidates, (entry_dist, entry_id))
        
        #Max-Heap: fartherst from candidates so we can remove it first
        found = []
        heapq.heappush(found, (-entry_dist, entry_id)) #negative to keep farthest on top

        
        visited = {entry_id}
        
        while candidates:
            #explore closest unexplored node
            current_dist, current_id = heapq.heappop(candidates)
            current_node = self.nodes[current_id]

            #stopping condition: farthest node
            worst_dist_found = -found[0][0]

            if current_dist > worst_dist_found:
                #if closes node to explore is farther than worst distance,
                #then we can stop, no neighbor will improve
                break

            for n_id in current_node.neighbors:

                if n_id in visited:
                    continue


                neigh_node = self.nodes[n_id]

                n_dist = self.distance_func(query, neigh_node.vector)
                
                visited.add(n_id)

                worst_dist_found = -found[0][0]

                #add if dist is smaller than worst found or if found list is not full yet
                if n_dist < worst_dist_found or len(found) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(found, (-n_dist, n_id))

                    #if found overfull, evict worst found
                    if len(found) > ef:
                        heapq.heappop(found)

        
        results = [(-neg_dist, id) for neg_dist, id in found]
        results = sorted(results)
        return results

    
    def _select_neighbors(self, candidates: List[tuple], M: int) -> List[Node]:

        #sort by distance
        candidates = sorted(candidates) #assume first in tuple is distance

        selected_ids = []

        if not candidates:
            return []
        
        #add first unconditionally
        selected_ids.append(candidates[0][1])

        #run diversity check for remaining candidates
        # if candidate C is closer to new node than to any other selected nodes - accept,
        # otherwise reject

        for dist_to_q, node_id in candidates[1:]:

            if len(selected_ids) >= M:
                break

            candidate_node = self.nodes[node_id]

            accept = True

            for selected_node_id in selected_ids:
                selected_node = self.nodes[selected_node_id]

                sel_to_candidate_d = self.distance_func(candidate_node.vector, selected_node.vector)

                if sel_to_candidate_d < dist_to_q:
                    
                    #don't take this node since it comes from direction we already have in selected nodes
                    accept = False
                    break
                
            
            if accept:
                selected_ids.append(node_id)

        selected_nodes = [self.nodes[id] for id in selected_ids]

        return selected_nodes




    

