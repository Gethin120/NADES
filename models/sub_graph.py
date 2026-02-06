import torch
import torch.nn.functional as F
import cugraph
import cudf
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph
from torch_geometric.transforms import RandomNodeSplit
from typing import List, Dict, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CugraphOverlapPartitioner:
    """
    Accelerated controlled-overlap graph partitioning using cuGraph.

    Method A: Louvain/Spectral + Expansion
    Method B: Louvain/1-Hop + Pruning
    Method C: Louvain community detection + direct subgraph construction
    Method D: CAOP (Controlled Overlap Partitioning) based on feature and degree similarity
    """

    def __init__(
        self,
        device=DEVICE,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
    ):
        """
        Initialize the partitioner.

        Args:
            device: Compute device
            train_ratio: Training split ratio (shared by all methods)
            val_ratio: Validation split ratio (shared by all methods)
        """
        self.device = device
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.min_community_size = 100
        if not torch.cuda.is_available():
            raise EnvironmentError("cuGraph requires a CUDA environment")

    def partition(
        self,
        method: str,
        data: Data,
        S: int = 8,
        s_max: int = 2,
        resolution: float = 1.0,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        Unified partitioning entrypoint; selects the strategy based on `method`.

        Args:
            method: Partition method. Supported: 'A', 'B', 'C', 'D'
                - 'A': Expansion
                - 'B': Pruning
                - 'C': Community Detection
                - 'D': CAOP
            data: Input graph data
            S: Target number of partitions (methods A/C/D; in C it is the macro-community count)
            s_max: Max number of overlapping communities per node (methods A/B/D)
            resolution: Louvain resolution parameter (method C)
            cache_path: Cache file path
            logger: Logger

        Returns:
            List of subgraphs
        """
        method = method.upper()

        if method == "A":
            return self.partition_expansion(data, S, s_max, cache_path, logger)
        elif method == "B":
            return self.partition_pruning(data, S, s_max, cache_path, logger)
        elif method == "C":
            return self.partition_community_detection(
                data, resolution, S_hint=S, cache_path=cache_path, logger=logger
            )
        elif method == "D":
            return self.partition_caop(data, S, s_max, cache_path, logger)
        else:
            raise ValueError(
                f"Unknown partition method '{method}'. Supported: 'A', 'B', 'C', 'D'"
            )

    def _pyg_to_cugraph_graph(self, data: Data) -> cugraph.Graph:
        """
        Convert a PyG `Data` object to a `cugraph.Graph`.

        Assumes `data.edge_index` is undirected and nodes are indexed 0..N-1.
        Ensures vertex type is int32 for `spectralBalancedCutClustering`.
        """
        # 1) Convert PyG edge_index (Tensor) to a cuDF DataFrame.
        #    cuGraph expects [src, dst] with int32 dtype.
        edge_index_cpu = data.edge_index.cpu().numpy()
        gdf = cudf.DataFrame()
        # Ensure int32 dtype (required by spectralBalancedCutClustering).
        gdf["src"] = edge_index_cpu[0].astype(np.int32)
        gdf["dst"] = edge_index_cpu[1].astype(np.int32)

        # 2) Create a cuGraph graph from cuDF edge list.
        # renumber=False assumes PyG node indices are already 0..N-1.
        G_cu = cugraph.Graph(directed=False)
        G_cu.from_cudf_edgelist(gdf, source="src", destination="dst", renumber=False)
        return G_cu

    def _load_cache(
        self, cache_path: Optional[str], logger=None
    ) -> Optional[List[Data]]:
        """
        Load subgraphs from cache (if available).

        Args:
            cache_path: Cache file path
            logger: Optional logger

        Returns:
            Cached list of subgraphs if present and valid; otherwise None.
        """
        if cache_path is None or not os.path.exists(cache_path):
            return None

        try:
            cached = torch.load(cache_path)
            if isinstance(cached, list):
                logger.info(f"Loaded subgraphs from cache: {len(cached)}")
                return cached
            logger.warning(
                f"Cache file {cache_path} has type {type(cached)}; rebuilding subgraphs."
            )
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}; recomputing.")

        return None

    def _save_cache(
        self, subgraphs: List[Data], cache_path: Optional[str], logger=None
    ) -> None:
        """
        Save subgraphs to cache.

        Args:
            subgraphs: List of subgraphs to save
            cache_path: Cache file path
            logger: Optional logger
        """
        if cache_path is None:
            return

        try:
            os.makedirs(
                os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
                exist_ok=True,
            )
            torch.save(subgraphs, cache_path)
            logger.info(f"Subgraphs saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _build_pyg_subgraphs(
        self,
        original_data: Data,
        node_to_community_map: List[Set[int]],
        S: int,
    ) -> List[Data]:
        """
        Build PyG `Data` subgraphs from the final node-to-community mapping.
        """

        # 1) Invert mapping: (node -> communities) -> (community -> nodes)
        # final_partitions[i] = list of node IDs that belong to community i
        final_partitions: List[List[int]] = [[] for _ in range(S)]
        for node_id, community_set in enumerate(node_to_community_map):
            for community_id in community_set:
                if community_id < S:  # Ensure bounds
                    final_partitions[community_id].append(node_id)

        # 2) Create a PyG `Data` object for each partition.
        subgraph_list: List[Data] = []
        # Keep tensors on the same device as the original graph.
        device = original_data.edge_index.device

        for community_id in range(S):
            original_node_indices = torch.tensor(
                final_partitions[community_id], dtype=torch.long, device=device
            )
            if original_node_indices.numel() == 0:
                continue  # Skip empty communities

            # 3) Extract edges with PyG's subgraph utility and relabel nodes.
            # relabel_nodes=True maps e.g. [10, 20, 30] -> [0, 1, 2].
            edge_index = original_data.edge_index.to(device)
            sub_edge_index, _ = pyg_subgraph(
                original_node_indices,
                edge_index,
                relabel_nodes=True,
                num_nodes=original_data.num_nodes,
            )

            # 4) Node features and labels
            sub_x = original_data.x[original_node_indices]
            sub_y = original_data.y[original_node_indices]

            # 5) Create the new Data object
            num_sub_nodes = sub_x.size(0)
            sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)
            sub_data.num_nodes = num_sub_nodes
            # Keep original node indices for downstream TSV computation.
            sub_data.original_node_indices = original_node_indices

            # 6) Create train/val/test masks with RandomNodeSplit.
            # Test ratio is the remainder.
            test_ratio = 1.0 - self.train_ratio - self.val_ratio
            splitter = RandomNodeSplit(
                split="train_rest",  # train is the remainder
                num_val=self.val_ratio,
                num_test=test_ratio,
                key="y",  # stratified by labels
            )
            sub_data = splitter(sub_data)

            # Ensure test_mask exists (if test_ratio == 0, create all-False).
            if not hasattr(sub_data, "test_mask") or sub_data.test_mask is None:
                sub_data.test_mask = torch.zeros(
                    num_sub_nodes, dtype=torch.bool, device=sub_data.x.device
                )

            # Validate subgraph: at least 2 nodes, has edges, has train nodes, and
            # train/val contain at least two classes.
            if sub_data.num_nodes < 2:
                continue  # Too few nodes

            if sub_data.edge_index.size(1) == 0:
                continue  # No edges

            if not hasattr(sub_data, "train_mask") or not sub_data.train_mask.any():
                continue  # No train nodes

            # Ensure training set has at least two classes (avoids issues in weight computation).
            train_labels = sub_data.y[sub_data.train_mask]
            unique_labels = torch.unique(train_labels)
            if len(unique_labels) < 2:
                continue  # Single-class train set
            # Ensure validation set has at least two classes
            val_labels = sub_data.y[sub_data.val_mask]
            unique_labels = torch.unique(val_labels)
            if len(unique_labels) < 2:
                continue  # Single-class val set

            subgraph_list.append(sub_data)

        return subgraph_list

    # --- Method A: cugraph.spectralBalancedCut + Expansion ---

    def partition_expansion(
        self,
        data: Data,
        S: int,
        s_max: int,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        Method A (Expansion):
        1) Use cugraph.spectralBalancedCutClustering to get a hard partition (s_max=1).
        2) Expand nodes to additional communities based on feature affinity until `s_max`.

        Args:
            data: Input graph data
            S: Target number of partitions
            s_max: Maximum overlap per node
            cache_path: Cache file path (disabled if None)
            logger: Optional logger
        """
        # Check cache
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"Starting [Method A: Expansion] (S={S}, s_max={s_max})")
        num_nodes = data.num_nodes

        # 1) Hard partitioning with cuGraph (initialization).
        # spectralBalancedCutClustering helps ensure we get exactly S partitions.
        G_cu = self._pyg_to_cugraph_graph(data)
        logger.info("... running cugraph.spectralBalancedCut ...")
        partition_df = cugraph.spectralBalancedCutClustering(G_cu, num_clusters=S)

        # 2) Build core communities and compute centroids.
        partition_pd = partition_df.to_pandas()
        partition_map = {
            int(row["vertex"]): int(row["cluster"])
            for _, row in partition_pd.iterrows()
        }

        core_communities: List[List[int]] = [[] for _ in range(S)]
        for node_id, part_id in partition_map.items():
            core_communities[part_id].append(node_id)

        x_cpu = data.x.cpu().numpy()
        centroids = np.array(
            [
                x_cpu[nodes].mean(axis=0)
                if len(nodes) > 0
                else np.zeros(x_cpu.shape[1])
                for nodes in core_communities
            ]
        )

        # 3) Expansion logic (similar to CAOP step 2).
        logger.info(f"... expanding {num_nodes} nodes up to s_max={s_max} ...")
        node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]
        node_counts = np.zeros(num_nodes, dtype=int)

        # 3a) Initialize: assign core communities
        for part_id, nodes in enumerate(core_communities):
            for node in nodes:
                node_to_community_map[node].add(part_id)
                node_counts[node] = 1

        # 3b) Greedy expansion: for each node, pick the top `s_max` communities by affinity
        for node in range(num_nodes):
            if node_counts[node] >= s_max:
                continue

            node_feature = x_cpu[node].reshape(1, -1)

            # Affinity to all S centroids
            affinities = cosine_similarity(node_feature, centroids)[0]

            # Sort community IDs by affinity (descending)
            sorted_community_ids = np.argsort(affinities)[::-1]

            # Add to the most similar communities until s_max
            for community_id in sorted_community_ids:
                if node_counts[node] >= s_max:
                    break
                # add() handles duplicates
                if community_id not in node_to_community_map[node]:
                    node_to_community_map[node].add(community_id)
                    node_counts[node] += 1

        logger.info("... expansion completed; building PyG subgraphs ...")
        subgraphs = self._build_pyg_subgraphs(data, node_to_community_map, S)

        # Save cache
        self._save_cache(subgraphs, cache_path, logger)

        return subgraphs

    # --- Method B: cugraph.louvain + 1-hop neighbors + Pruning ---

    def partition_pruning(
        self,
        data: Data,
        S_hint: int,
        s_max: int,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        Method B (Pruning):
        1) Use cugraph.louvain to find S' "natural" communities.
        2) Create "uncontrolled" overlap via 1-hop neighbors (to mimic SLLP).
        3) Prune extra communities based on feature affinity until `s_max`.

        Note: `S_hint` is ignored in this method; the final community count is determined by Louvain.

        Args:
            data: Input graph data
            S_hint: Suggested number of communities (ignored)
            s_max: Maximum overlap per node
            cache_path: Cache file path (disabled if None)
            logger: Optional logger
        """
        # Check cache
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"Starting [Method B: Pruning] (s_max={s_max})")
        num_nodes = data.num_nodes

        # 1) Hard partitioning with cuGraph (Louvain)
        G_cu = self._pyg_to_cugraph_graph(data)
        logger.info("... running cugraph.louvain ...")
        partition_df = cugraph.louvain(G_cu)
        partition_df = partition_df[0]
        partition_pd = partition_df.to_pandas()

        partition_map = {row.vertex: row.partition for row in partition_pd.itertuples()}
        S_found = partition_pd["partition"].max() + 1
        logger.info(f"... Louvain found {S_found} natural communities ...")

        # 2) Create 1-hop neighbor overlap (uncontrolled overlap, mimicking SLLP)
        logger.info("... creating 1-hop neighbor overlap ...")
        adj: Dict[int, Set[int]] = {}
        src, dst = data.edge_index.cpu().numpy()
        for i in range(len(src)):
            u, v = src[i], dst[i]
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        initial_node_to_community_map: List[Set[int]] = [
            set() for _ in range(num_nodes)
        ]
        max_overlap_found = 0
        for node in range(num_nodes):
            # Add the node's own community
            if node in partition_map:
                initial_node_to_community_map[node].add(partition_map[node])
            # Add all 1-hop neighbors' communities
            for neighbor in adj.get(node, set()):
                if neighbor in partition_map:
                    initial_node_to_community_map[node].add(partition_map[neighbor])

            if len(initial_node_to_community_map[node]) > max_overlap_found:
                max_overlap_found = len(initial_node_to_community_map[node])

        logger.info(f"... maximum uncontrolled overlap found: s = {max_overlap_found} ...")

        # 3) Pruning logic
        logger.info(f"... pruning {num_nodes} nodes to s_max={s_max} ...")
        x_cpu = data.x.cpu().numpy()

        # 3a) Compute centroids for all S_found communities
        core_communities: List[List[int]] = [[] for _ in range(S_found)]
        for node_id, part_id in partition_map.items():
            core_communities[part_id].append(node_id)

        centroids = np.array(
            [
                x_cpu[nodes].mean(axis=0)
                if len(nodes) > 0
                else np.zeros(x_cpu.shape[1])
                for nodes in core_communities
            ]
        )

        # 3b) Prune
        final_node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]

        for node in range(num_nodes):
            communities = list(initial_node_to_community_map[node])

            if len(communities) == 0:
                continue
            elif len(communities) <= s_max:
                final_node_to_community_map[node] = set(communities)
            else:
                # Affinity computation and pruning
                node_feature = x_cpu[node].reshape(1, -1)

                # Centroids for the communities this node belongs to
                community_centroids = centroids[communities]

                # Affinity to these community centroids
                affinities = cosine_similarity(node_feature, community_centroids)[0]

                # Sort local indices by affinity (descending)
                sorted_local_indices = np.argsort(affinities)[::-1]

                # Keep the top s_max communities
                for i in range(s_max):
                    local_idx = sorted_local_indices[i]
                    global_community_id = communities[local_idx]
                    final_node_to_community_map[node].add(global_community_id)

        logger.info("... pruning completed; building PyG subgraphs ...")
        subgraphs = self._build_pyg_subgraphs(
            data, final_node_to_community_map, S_found
        )

        # Filter subgraphs smaller than min_community_size
        filtered_subgraphs = [
            sg for sg in subgraphs if sg.num_nodes >= self.min_community_size
        ]
        num_filtered = len(subgraphs) - len(filtered_subgraphs)

        if num_filtered > 0:
            logger.info(
                f"Filtered out {num_filtered} subgraphs with fewer than {self.min_community_size} nodes"
            )

        if len(filtered_subgraphs) == 0:
            error_msg = (
                f"No subgraphs remain after filtering (all subgraphs have fewer than {self.min_community_size} nodes)"
            )
            logger.error(error_msg)
            return []

        # Save cache
        self._save_cache(filtered_subgraphs, cache_path, logger)

        return filtered_subgraphs

    # --- Method C: cugraph.louvain + direct subgraph construction ---

    def partition_community_detection(
        self,
        data: Data,
        resolution: float = 1.0,
        S_hint: Optional[int] = None,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        Method C (Community Detection):
        1) Run community detection using cugraph.louvain (with optional resolution).
        2) Build a subgraph for each community without expansion or pruning.
        3) Optionally supports on-disk caching.

        Args:
            data: Input graph data
            resolution: Louvain resolution parameter controlling community sizes
            S_hint: If provided, merge natural communities into `S_hint` macro-communities via modulo
            cache_path: Cache file path (disabled if None)
            logger: Optional logger

        Returns:
            List of subgraphs
        """
        # Check cache
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"Starting [Method C: Community Detection] (resolution={resolution})")
        num_nodes = data.num_nodes

        # 1) Run Louvain community detection
        G_cu = self._pyg_to_cugraph_graph(data)
        logger.info("... running cugraph.louvain ...")
        communities: Dict[int, List[int]] = {}

        try:
            # Try using the resolution parameter
            try:
                parts_df, _ = cugraph.louvain(G_cu, resolution=resolution)
            except (TypeError, ValueError):
                # If resolution is unsupported, fall back to default call.
                logger.info("cugraph.louvain does not support `resolution`; using default call")
                parts_df = cugraph.louvain(G_cu)

            # Convert to pandas
            if isinstance(parts_df, tuple):
                parts_df = parts_df[0]
            parts_pd = parts_df.to_pandas()

            for _, row in parts_pd.iterrows():
                node_id = int(row["vertex"])
                partition_id = int(row["partition"])
                communities.setdefault(partition_id, []).append(node_id)

        except Exception as e:
            logger.warning(f"Louvain community detection failed ({e}); falling back to single-node communities.")

        if not communities:
            logger.warning("Community detection produced no valid partition; falling back to single-node communities.")
            all_nodes = torch.arange(num_nodes, device=self.device)
            communities = {
                idx: [int(node)] for idx, node in enumerate(all_nodes.cpu().tolist())
            }

        num_communities = len(communities)
        logger.info(f"... Louvain detected {num_communities} communities ...")

        # Community sizes
        community_sizes = {
            comm_id: len(nodes) for comm_id, nodes in communities.items()
        }
        # sorted_comm_ids = sorted(community_sizes.keys())
        # logger.info("... Community size per community:")
        # for comm_id in sorted_comm_ids:
        #     size = community_sizes[comm_id]
        #     logger.info(f"  community {comm_id}: {size} nodes")

        # Summary stats
        sizes_list = list(community_sizes.values())
        if sizes_list:
            min_size = min(sizes_list)
            max_size = max(sizes_list)
            avg_size = sum(sizes_list) / len(sizes_list)
            logger.info(
                f"  Community size stats: min={min_size}, max={max_size}, avg={avg_size:.1f}"
            )

        # Filter out communities smaller than min_community_size
        filtered_communities = {
            comm_id: nodes
            for comm_id, nodes in communities.items()
            if len(nodes) >= self.min_community_size
        }
        num_filtered = len(communities) - len(filtered_communities)

        if num_filtered > 0:
            logger.info(
                f"Filtered out {num_filtered} communities with fewer than {self.min_community_size} nodes"
            )
            # List filtered communities (truncated)
            filtered_sizes = [
                (comm_id, len(nodes))
                for comm_id, nodes in communities.items()
                if len(nodes) < self.min_community_size
            ]
            if filtered_sizes:
                filtered_info = ", ".join(
                    [
                        f"comm{comm_id}({size} nodes)"
                        for comm_id, size in filtered_sizes[:10]
                    ]
                )
                if len(filtered_sizes) > 10:
                    filtered_info += f" ... (total {len(filtered_sizes)})"
                logger.info(f"  Filtered communities: {filtered_info}")

        communities = filtered_communities
        num_communities = len(communities)

        # If S_hint is set, merge into macro-communities after filtering
        if S_hint is not None:
            if S_hint <= 0:
                raise ValueError("S_hint must be a positive integer")
            logger.info(f"... merging communities into {S_hint} macro-communities (modulo) ...")
            macro_communities = {i: [] for i in range(S_hint)}
            for comm_id, nodes in communities.items():
                macro_id = comm_id % S_hint
                macro_communities[macro_id].extend(nodes)
            communities = {k: v for k, v in macro_communities.items() if len(v) > 0}
            num_communities = len(communities)
            logger.info(
                "Macro-community sizes: "
                + ", ".join([f"Macro {k}: {len(v)}" for k, v in communities.items()])
            )

        if num_communities == 0:
            error_msg = (
                f"No communities remain after filtering (all communities have fewer than {self.min_community_size} nodes)"
            )
            logger.error(error_msg)
            # Return empty list.
            return []

        logger.info(f"{num_communities} communities remain after filtering")

        # 2) Convert community mapping into node->community mapping (single membership)
        node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]
        for comm_id, nodes in communities.items():
            for node_id in nodes:
                if node_id < num_nodes:
                    node_to_community_map[node_id].add(comm_id)

        # 3) Build subgraphs
        logger.info("... building subgraphs via _build_pyg_subgraphs ...")
        subgraphs = self._build_pyg_subgraphs(
            data, node_to_community_map, num_communities
        )

        logger.info(f"... completed; built {len(subgraphs)} subgraphs ...")

        # 4) Save cache
        self._save_cache(subgraphs, cache_path, logger)

        return subgraphs

    # --- Method D: CAOP (Controlled Overlap Partitioning) ---

    def _degree_vector(self, data: Data) -> torch.Tensor:
        """Compute the node degree vector."""
        src = data.edge_index[0]
        deg = torch.bincount(src, minlength=data.num_nodes)
        return deg

    def _core_priority_pruning(
        self,
        macro_communities: List[Set[int]],
        candidate_subgraphs: List[Set[int]],
        data: Data,
        logger=None,
    ) -> List[Set[int]]:
        """
        "Core-first" pruning: under the s_max=1 constraint, prefer keeping core
        nodes in each macro-community.

        Args:
            macro_communities: Macro-community list; each element is the set of core node IDs
            candidate_subgraphs: Candidate subgraphs (macro-community nodes + 1-hop neighbors)
            data: Input graph data
            logger: Optional logger

        Returns:
            Pruned subgraphs
        """
        x_cpu = data.x.cpu().numpy()
        num_subgraphs = len(candidate_subgraphs)

        # Compute centroids for each subgraph (for similarity-based arbitration)
        subgraph_centroids = []
        for sg_nodes in candidate_subgraphs:
            if len(sg_nodes) > 0:
                nodes_list = list(sg_nodes)
                centroid = x_cpu[nodes_list].mean(axis=0)
                subgraph_centroids.append(centroid)
            else:
                subgraph_centroids.append(np.zeros(x_cpu.shape[1]))

        # Track which subgraph each node is assigned to
        node_to_subgraph: Dict[int, int] = {}
        final_subgraphs: List[Set[int]] = [set() for _ in range(num_subgraphs)]

        # Pass 1: core nodes (assigned with highest priority)
        for subgraph_id in range(num_subgraphs):
            core_nodes = macro_communities[subgraph_id]
            for node_id in core_nodes:
                if node_id not in node_to_subgraph:
                    # Core node not yet assigned -> assign to its macro-community
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                else:
                    # Core node already assigned -> arbitrate by similarity
                    old_subgraph_id = node_to_subgraph[node_id]
                    node_feature = x_cpu[node_id].reshape(1, -1)

                    # Similarity to old vs new subgraph centroids
                    old_centroid = subgraph_centroids[old_subgraph_id].reshape(1, -1)
                    new_centroid = subgraph_centroids[subgraph_id].reshape(1, -1)

                    old_sim = cosine_similarity(node_feature, old_centroid)[0, 0]
                    new_sim = cosine_similarity(node_feature, new_centroid)[0, 0]

                    # If the new subgraph is more similar, reassign.
                    if new_sim > old_sim:
                        final_subgraphs[old_subgraph_id].remove(node_id)
                        node_to_subgraph[node_id] = subgraph_id
                        final_subgraphs[subgraph_id].add(node_id)

        # Pass 2: non-core nodes (1-hop neighbors)
        for subgraph_id in range(num_subgraphs):
            core_nodes = macro_communities[subgraph_id]
            candidate_nodes = candidate_subgraphs[subgraph_id]
            non_core_nodes = candidate_nodes - core_nodes

            for node_id in non_core_nodes:
                if node_id not in node_to_subgraph:
                    # Non-core node not yet assigned -> add it
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                # If already in another subgraph, do not add (strict s_max=1).

        return final_subgraphs

    def _core_priority_pruning_with_fraud(
        self,
        macro_communities: List[Set[int]],
        candidate_subgraphs: List[Set[int]],
        data: Data,
        fraud_weight: float = 0.5,
        logger=None,
    ) -> List[Set[int]]:
        """
        "Core-first + fraud-first" pruning: under s_max=1, prefer keeping core nodes
        and prioritize fraud-labeled nodes when conflicts arise.

        Priority:
        1) Core fraud nodes (in macro-community and labeled fraud)
        2) Non-core fraud nodes (fraud nodes among 1-hop neighbors)
        3) Normal nodes

        Args:
            macro_communities: Macro-community list; each element is the set of core node IDs
            candidate_subgraphs: Candidate subgraphs (macro-community nodes + 1-hop neighbors)
            data: Input graph data (must contain `y` labels)
            fraud_weight: Additional weight for fraud nodes (used to adjust similarity)
            logger: Optional logger

        Returns:
            Pruned subgraphs
        """
        x_cpu = data.x.cpu().numpy()
        num_subgraphs = len(candidate_subgraphs)

        # Fraud labels (assume fraud=1, normal=0)
        if not hasattr(data, "y") or data.y is None:
            logger.warning("No labels found in data; falling back to core-first pruning")
            return self._core_priority_pruning(
                macro_communities, candidate_subgraphs, data, logger
            )

        y_cpu = data.y.cpu().numpy()
        # Binary assumption: 1=fraud, 0=normal (adjust for multi-class as needed)
        if y_cpu.dtype in [np.int64, np.int32]:
            fraud_labels = (y_cpu == 1).astype(np.float32)
        else:
            # If float labels, treat > 0.5 as fraud.
            fraud_labels = (y_cpu > 0.5).astype(np.float32)

        # Centroids for each subgraph (for similarity-based arbitration)
        subgraph_centroids = []
        for sg_nodes in candidate_subgraphs:
            if len(sg_nodes) > 0:
                nodes_list = list(sg_nodes)
                centroid = x_cpu[nodes_list].mean(axis=0)
                subgraph_centroids.append(centroid)
            else:
                subgraph_centroids.append(np.zeros(x_cpu.shape[1]))

        # Track assignment of each node to a subgraph
        node_to_subgraph: Dict[int, int] = {}
        final_subgraphs: List[Set[int]] = [set() for _ in range(num_subgraphs)]

        # Pass 1: core nodes (fraud-first within core)
        for subgraph_id in range(num_subgraphs):
            core_nodes = macro_communities[subgraph_id]
            # Split core fraud vs core normal nodes
            core_fraud_nodes = [
                node
                for node in core_nodes
                if node < len(fraud_labels) and fraud_labels[node] > 0.5
            ]
            core_normal_nodes = [
                node
                for node in core_nodes
                if node < len(fraud_labels) and fraud_labels[node] <= 0.5
            ]

            # Core fraud nodes first
            for node_id in core_fraud_nodes:
                if node_id not in node_to_subgraph:
                    # Unassigned core fraud node -> assign directly
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                else:
                    # Already assigned -> arbitrate by similarity
                    old_subgraph_id = node_to_subgraph[node_id]
                    node_feature = x_cpu[node_id].reshape(1, -1)

                    # Similarity to old vs new subgraph centroids
                    old_centroid = subgraph_centroids[old_subgraph_id].reshape(1, -1)
                    new_centroid = subgraph_centroids[subgraph_id].reshape(1, -1)

                    old_sim = cosine_similarity(node_feature, old_centroid)[0, 0]
                    new_sim = cosine_similarity(node_feature, new_centroid)[0, 0]

                    # For core fraud nodes, reassign if the new subgraph is more similar
                    if new_sim > old_sim:
                        final_subgraphs[old_subgraph_id].remove(node_id)
                        node_to_subgraph[node_id] = subgraph_id
                        final_subgraphs[subgraph_id].add(node_id)

            # Core normal nodes
            for node_id in core_normal_nodes:
                if node_id not in node_to_subgraph:
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                else:
                    # Already assigned -> arbitrate by similarity
                    old_subgraph_id = node_to_subgraph[node_id]
                    node_feature = x_cpu[node_id].reshape(1, -1)

                    old_centroid = subgraph_centroids[old_subgraph_id].reshape(1, -1)
                    new_centroid = subgraph_centroids[subgraph_id].reshape(1, -1)

                    old_sim = cosine_similarity(node_feature, old_centroid)[0, 0]
                    new_sim = cosine_similarity(node_feature, new_centroid)[0, 0]

                    if new_sim > old_sim:
                        final_subgraphs[old_subgraph_id].remove(node_id)
                        node_to_subgraph[node_id] = subgraph_id
                        final_subgraphs[subgraph_id].add(node_id)

        # Pass 2: non-core nodes (1-hop neighbors), fraud-first
        for subgraph_id in range(num_subgraphs):
            core_nodes = macro_communities[subgraph_id]
            candidate_nodes = candidate_subgraphs[subgraph_id]
            non_core_nodes = candidate_nodes - core_nodes

            # Split non-core fraud vs normal nodes
            non_core_fraud_nodes = [
                node
                for node in non_core_nodes
                if node < len(fraud_labels) and fraud_labels[node] > 0.5
            ]
            non_core_normal_nodes = [
                node
                for node in non_core_nodes
                if node < len(fraud_labels) and fraud_labels[node] <= 0.5
            ]

            # Non-core fraud nodes first
            for node_id in non_core_fraud_nodes:
                if node_id not in node_to_subgraph:
                    # Unassigned -> add
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                else:
                    # Already assigned -> arbitrate by weighted similarity
                    old_subgraph_id = node_to_subgraph[node_id]
                    node_feature = x_cpu[node_id].reshape(1, -1)

                    old_centroid = subgraph_centroids[old_subgraph_id].reshape(1, -1)
                    new_centroid = subgraph_centroids[subgraph_id].reshape(1, -1)

                    old_sim = cosine_similarity(node_feature, old_centroid)[0, 0]
                    new_sim = cosine_similarity(node_feature, new_centroid)[0, 0]

                    # Fraud weighting: favor the subgraph that contains more fraud nodes
                    old_subgraph_fraud_count = sum(
                        1
                        for n in final_subgraphs[old_subgraph_id]
                        if n < len(fraud_labels) and fraud_labels[n] > 0.5
                    )
                    new_subgraph_fraud_count = sum(
                        1
                        for n in final_subgraphs[subgraph_id]
                        if n < len(fraud_labels) and fraud_labels[n] > 0.5
                    )

                    # Weighted similarity: if the new subgraph has more fraud nodes, add a bonus
                    old_weighted_sim = old_sim + (
                        fraud_weight
                        * old_subgraph_fraud_count
                        / max(1, len(final_subgraphs[old_subgraph_id]))
                    )
                    new_weighted_sim = new_sim + (
                        fraud_weight
                        * new_subgraph_fraud_count
                        / max(1, len(final_subgraphs[subgraph_id]))
                    )

                    if new_weighted_sim > old_weighted_sim:
                        final_subgraphs[old_subgraph_id].remove(node_id)
                        node_to_subgraph[node_id] = subgraph_id
                        final_subgraphs[subgraph_id].add(node_id)

            # Non-core normal nodes (lowest priority)
            for node_id in non_core_normal_nodes:
                if node_id not in node_to_subgraph:
                    # Unassigned -> add
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                # If already assigned elsewhere, do not add (strict s_max=1).

        return final_subgraphs

    # --- Method D: CAOP ---

    def partition_caop(
        self,
        data: Data,
        S: int,
        s_max: int,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        Method D (CAOP):
        Controlled-overlap graph partitioning based on feature similarity and degree similarity.

        Args:
            data: Input graph data
            S: Target number of partitions
            s_max: Maximum overlap per node
            cache_path: Cache file path (disabled if None)
            logger: Optional logger

        Returns:
            List of subgraphs
        """
        # Check cache
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"Starting [Method D: CAOP] (S={S}, s_max={s_max})")
        device = self.device
        x = data.x.to(device)
        n = data.num_nodes
        deg = self._degree_vector(data).float().to(device)
        max_deg = deg.max().clamp_min(1.0)
        x = F.normalize(x, dim=1)

        # 1) Select seed nodes
        rng = torch.Generator(device="cpu")
        first = int(torch.randint(low=0, high=n, size=(1,), generator=rng).item())
        seeds = [first]
        sub_ratio = 0.2 if n > 10000 else 1.0
        sub_idx = torch.arange(n)
        if sub_ratio < 1.0:
            k = max(1, int(n * sub_ratio))
            sub_idx = sub_idx[torch.randperm(n, generator=rng)[:k]]
        x_sub = x[sub_idx]
        dists = (1.0 - (x_sub @ x[first].unsqueeze(1)).squeeze()).clamp_min(0.0)

        for _ in range(1, S):
            probs = (dists + 1e-6) / (dists.sum() + 1e-6)
            pick_pos = torch.multinomial(probs, num_samples=1, replacement=False).item()
            pick = int(sub_idx[pick_pos].item())
            seeds.append(pick)
            new_d = (1.0 - (x_sub @ x[pick].unsqueeze(1)).squeeze()).clamp_min(0)
            dists = torch.minimum(dists, new_d)

        # 2) Initialize groups
        groups: List[List[int]] = [[s] for s in seeds]
        assigned_counts = torch.zeros(n, dtype=torch.long, device=device)
        assigned_counts[torch.tensor(seeds, device=device)] = 1
        group_feat_sum = x[torch.tensor(seeds, device=device)].clone()
        group_deg_sum = deg[torch.tensor(seeds, device=device)].clone()
        group_counts = torch.ones(S, device=device)

        # 3) Greedy assignment of nodes to communities
        K = min(8, S)
        batch_size = 20000 if n > 50000 else 10000
        logger.info(f"... assigning {n} nodes to {S} communities (s_max={s_max}) ...")

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            idx = torch.arange(start, end, device=device)
            mask_remain = assigned_counts[idx] < s_max
            if not mask_remain.any():
                continue
            idx = idx[mask_remain]
            g_feat_mean = F.normalize(group_feat_sum / group_counts.unsqueeze(1), dim=1)
            g_deg_mean = (group_deg_sum / group_counts).unsqueeze(0)
            sim_f = (x[idx] @ g_feat_mean.t()).clamp(-1, 1)
            sim_d = 1.0 - (deg[idx].unsqueeze(1) - g_deg_mean).abs() / max_deg
            affinity = sim_f + sim_d
            _, topk_idx = torch.topk(affinity, k=K, dim=1)
            for row, node in enumerate(idx.tolist()):
                if assigned_counts[node] >= s_max:
                    continue
                cands = topk_idx[row]
                for gi in cands.tolist():
                    if assigned_counts[node] >= s_max:
                        break
                    groups[gi].append(node)
                    assigned_counts[node] += 1
                    group_feat_sum[gi] += x[node]
                    group_deg_sum[gi] += deg[node]
                    group_counts[gi] += 1

        # 4) Build subgraphs
        logger.info("... building subgraphs ...")
        subgraphs = []
        edge_device = data.edge_index.device

        for nodes in groups:
            if not nodes:
                continue

            # Convert to a sorted node-index tensor
            node_indices = torch.tensor(
                sorted(set(int(i) for i in nodes)), dtype=torch.long, device=edge_device
            )

            # Extract the subgraph using PyG's subgraph utility
            sub_edge_index, edge_mask = pyg_subgraph(
                node_indices,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )

            # Create subgraph data object
            d = Data(
                x=data.x[node_indices],
                edge_index=sub_edge_index,
                y=data.y[node_indices],
            )
            d.num_nodes = node_indices.numel()
            # Keep original node indices for downstream TSV computation
            d.original_node_indices = node_indices

            # Create train/val/test masks
            test_ratio = 1.0 - self.train_ratio - self.val_ratio
            splitter = RandomNodeSplit(
                split="train_rest",
                num_val=self.val_ratio,
                num_test=test_ratio,
                key="y",
            )
            d = splitter(d)
            d.test_mask = torch.zeros(d.num_nodes, dtype=torch.bool, device=d.x.device)

            subgraphs.append(d)
        subgraphs = self._filter_label_balanced_subgraphs(
            subgraphs, logger
        )  # Not needed for yelp/amazon; required for elliptic.
        logger.info(f"... completed; built {len(subgraphs)} subgraphs ...")

        # 5) Save cache
        self._save_cache(subgraphs, cache_path, logger)

        return subgraphs



    # --- Method G: global optimal pruning (two-stage) ---

    def _generate_d_candidates(
        self,
        data: Data,
        N_D: int = 20,
        s_max_D_temp: int = 4,
        logger=None,
    ) -> List[Set[int]]:
        """
        Generate type-D candidate subgraphs (based on CAOP with a larger temporary s_max).

        Args:
            data: Input graph data
            N_D: Number of candidate subgraphs to generate
            s_max_D_temp: Temporary s_max used for candidate generation
            logger: Optional logger

        Returns:
            Candidate subgraphs as a list of node-id sets.
        """
        logger.info(f"Generating type-D candidates (N_D={N_D}, s_max_D_temp={s_max_D_temp})...")

        # Generate candidates via CAOP (with relaxed s_max)
        caop_subgraphs = self.partition_caop(
            data, S=N_D, s_max=s_max_D_temp, cache_path=None, logger=logger
        )

        # Convert to list of node-id sets
        d_candidates = []
        for sg in caop_subgraphs:
            if hasattr(sg, "original_node_indices"):
                node_set = set(sg.original_node_indices.cpu().tolist())
            else:
                node_set = set(range(sg.num_nodes))
            if len(node_set) > 0:
                d_candidates.append(node_set)

        logger.info(f"Generated {len(d_candidates)} type-D candidate subgraphs")

        return d_candidates

    def _generate_e_candidates(
        self,
        data: Data,
        N_E: int = 20,
        logger=None,
    ) -> List[Set[int]]:
        """
        Generate type-E candidate subgraphs (E-style construction without enforcing s_max).

        Args:
            data: Input graph data
            N_E: Number of candidates to generate (used as macro-community count)
            logger: Optional logger

        Returns:
            Candidate subgraphs as a list of node-id sets.
        """
        logger.info(f"Generating type-E candidates (N_E={N_E})...")

        # 1) Louvain community discovery
        G_cu = self._pyg_to_cugraph_graph(data)
        partition_df = cugraph.louvain(G_cu)
        if isinstance(partition_df, tuple):
            partition_df = partition_df[0]
        partition_pd = partition_df.to_pandas()

        partition_map = {row.vertex: row.partition for row in partition_pd.itertuples()}

        # 2) Filter out small communities
        communities: Dict[int, List[int]] = {}
        for node_id, part_id in partition_map.items():
            communities.setdefault(part_id, []).append(node_id)

        filtered_communities = {
            comm_id: nodes
            for comm_id, nodes in communities.items()
            if len(nodes) >= self.min_community_size
        }

        if len(filtered_communities) == 0:
            logger.warning("No communities remain after filtering")
            return []

        # 3) Merge into N_E macro-communities (modulo)
        macro_communities: Dict[int, List[int]] = {i: [] for i in range(N_E)}
        for comm_id, nodes in filtered_communities.items():
            macro_id = comm_id % N_E
            macro_communities[macro_id].extend(nodes)

        # 4) 1-hop expansion (no s_max limit)
        adj: Dict[int, Set[int]] = {}
        src, dst = data.edge_index.cpu().numpy()
        for i in range(len(src)):
            u, v = src[i], dst[i]
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        e_candidates = []
        for macro_id, macro_nodes in macro_communities.items():
            if len(macro_nodes) == 0:
                continue
            # Expand 1-hop neighbors
            expanded_nodes = set(macro_nodes)
            for node in macro_nodes:
                for neighbor in adj.get(node, set()):
                    expanded_nodes.add(neighbor)
            if len(expanded_nodes) > 0:
                e_candidates.append(expanded_nodes)

        logger.info(f"Generated {len(e_candidates)} type-E candidate subgraphs")

        return e_candidates

    def _global_optimal_pruning(
        self,
        candidate_pool: List[Set[int]],
        data: Data,
        max_overlap_per_node: int = 3,
        min_subgraph_size: int = 10,
        logger=None,
    ) -> List[Set[int]]:
        """
        Global optimal pruning: select subgraphs from a candidate pool under a global
        `max_overlap_per_node` constraint.

        Args:
            candidate_pool: Candidate pool (each element is a set of node IDs)
            data: Input graph data
            max_overlap_per_node: Max overlap per node (global constraint)
            min_subgraph_size: Minimum subgraph size
            logger: Optional logger

        Returns:
            Pruned subgraphs (as node-id sets)
        """
        logger.info(
            f"Starting global optimal pruning (candidates={len(candidate_pool)}, max_overlap={max_overlap_per_node})..."
        )

        num_nodes = data.num_nodes

        # Fraud labels (optional)
        fraud_labels = None
        if hasattr(data, "y") and data.y is not None:
            y_cpu = data.y.cpu().numpy()
            if y_cpu.dtype in [np.int64, np.int32]:
                fraud_labels = (y_cpu == 1).astype(np.float32)
            else:
                fraud_labels = (y_cpu > 0.5).astype(np.float32)

        # 1) Sort candidates (fraud ratio, size, etc.)
        def score_subgraph(subgraph_nodes: Set[int]) -> float:
            """Compute a score for candidate sorting."""
            if len(subgraph_nodes) == 0:
                return 0.0

            # Base score: subgraph size
            size_score = len(subgraph_nodes) / num_nodes

            # Fraud ratio score
            fraud_score = 0.0
            if fraud_labels is not None:
                fraud_count = sum(
                    1
                    for node in subgraph_nodes
                    if node < len(fraud_labels) and fraud_labels[node] > 0.5
                )
                fraud_score = (
                    fraud_count / len(subgraph_nodes)
                    if len(subgraph_nodes) > 0
                    else 0.0
                )

            # Combined score: put more weight on fraud ratio
            return fraud_score * 0.7 + size_score * 0.3

        sorted_candidates = sorted(candidate_pool, key=score_subgraph, reverse=True)

        # 2) Iterative greedy allocation
        final_subgraphs: List[Set[int]] = []
        global_overlap_counts = np.zeros(num_nodes, dtype=np.int32)

        # Track core nodes (ideally seeds from CAOP and macro-community centers from E)
        core_nodes_set = set()
        for candidate in candidate_pool:
            # Simplified: treat the first node as core (ideally derive from CAOP/E internals)
            if len(candidate) > 0:
                core_nodes_set.add(list(candidate)[0])

        for candidate in sorted_candidates:
            # Build a temporary subgraph
            temp_subgraph = set(candidate)

            # Filter nodes that already reached max_overlap
            nodes_to_remove = []
            for node_id in temp_subgraph:
                if node_id >= num_nodes:
                    nodes_to_remove.append(node_id)
                elif global_overlap_counts[node_id] >= max_overlap_per_node:
                    nodes_to_remove.append(node_id)

            for node_id in nodes_to_remove:
                temp_subgraph.remove(node_id)

            # Drop if too small after filtering
            if len(temp_subgraph) < min_subgraph_size:
                continue

            # Special handling: prioritize fraud nodes and core nodes when budget allows.
            priority_nodes = []
            normal_nodes = []

            for node_id in temp_subgraph:
                if node_id >= num_nodes:
                    continue

                is_fraud = (
                    fraud_labels is not None
                    and node_id < len(fraud_labels)
                    and fraud_labels[node_id] > 0.5
                )
                is_core = node_id in core_nodes_set
                overlap_count = global_overlap_counts[node_id]

                # Fraud/core nodes with remaining budget
                if (is_fraud and overlap_count < max_overlap_per_node) or (
                    is_core and overlap_count < max_overlap_per_node
                ):
                    priority_nodes.append(node_id)
                else:
                    normal_nodes.append(node_id)

            # Consume budget for priority nodes first
            for node_id in priority_nodes:
                if global_overlap_counts[node_id] < max_overlap_per_node:
                    global_overlap_counts[node_id] += 1

            # Then consume budget for normal nodes
            for node_id in normal_nodes:
                if global_overlap_counts[node_id] < max_overlap_per_node:
                    global_overlap_counts[node_id] += 1

            # Keep only nodes that were actually admitted under budget
            final_temp_subgraph = {
                node_id
                for node_id in temp_subgraph
                if node_id < num_nodes
                and global_overlap_counts[node_id] > 0
                and node_id in priority_nodes + normal_nodes
            }

            if len(final_temp_subgraph) >= min_subgraph_size:
                final_subgraphs.append(final_temp_subgraph)

        logger.info(
            f"Global pruning completed: selected {len(final_subgraphs)} subgraphs from {len(candidate_pool)} candidates"
        )

        return final_subgraphs


    def _filter_label_balanced_subgraphs(
        self, subgraphs: List[Data], logger=None
    ) -> List[Data]:
        """
        Filter subgraphs with the following requirements:
            1) `num_nodes` >= `self.min_community_size`
            2) Contains at least one edge
            3) Overall `y` contains both 0 and 1
            4) `train_mask` and `val_mask` each contain both 0 and 1
        """

        def _has_both(label_tensor: torch.Tensor, mask: Optional[torch.Tensor] = None):
            if label_tensor is None:
                return False
            labels = label_tensor.view(-1)
            if mask is not None:
                mask = mask.view(-1)
                if mask.numel() != labels.numel():
                    mask = mask[: labels.numel()]
                if not mask.any():
                    return False
                mask = mask.to(labels.device)
                labels = labels[mask]
            return (labels == 0).any().item() and (labels == 1).any().item()

        balanced = []
        removed_small = 0
        removed_edge = 0
        removed_label = 0
        removed_train = 0
        removed_val = 0

        for sg in subgraphs:
            if sg.num_nodes < self.min_community_size:
                removed_small += 1
                continue
            if sg.edge_index is None or sg.edge_index.size(1) == 0:
                removed_edge += 1
                continue
            y = sg.y
            if y is None or not _has_both(y):
                removed_label += 1
                continue
            train_mask = getattr(sg, "train_mask", None)
            val_mask = getattr(sg, "val_mask", None)
            if train_mask is None or not _has_both(y, train_mask):
                removed_train += 1
                continue
            if val_mask is None or not _has_both(y, val_mask):
                removed_val += 1
                continue
            balanced.append(sg)

        if logger is not None:
            if removed_small > 0:
                logger.info(
                    f"Filtered out {removed_small} subgraphs with fewer than {self.min_community_size} nodes"
                )
            if removed_edge > 0:
                logger.info(f"Filtered out {removed_edge} subgraphs with no edges")
            if removed_label > 0:
                logger.info(
                    f"Filtered out {removed_label} subgraphs whose overall labels do not contain both 0 and 1"
                )
            if removed_train > 0:
                logger.info(
                    f"Filtered out {removed_train} subgraphs whose train split labels do not contain both 0 and 1"
                )
            if removed_val > 0:
                logger.info(
                    f"Filtered out {removed_val} subgraphs whose val split labels do not contain both 0 and 1"
                )

        return balanced

    def compute_tsv(
        self,
        private_data: Data,
        subgraphs: List[Data],
        cache_path: Optional[str] = None,
        logger=None,
    ) -> torch.Tensor:
        """
        Compute TSV (Task-Specific Vector) features.

        Args:
            private_data: Private graph data
            subgraphs: Subgraph list
            cache_path: Cache file path (disabled if None)
            logger: Optional logger

        Returns:
            TSV tensor with shape [num_subgraphs, 6]
        """
        num_subgraphs = len(subgraphs)
        device = self.device

        # Check cache
        if cache_path is not None and os.path.exists(cache_path):
            try:
                cached_tsv = torch.load(cache_path)
                # Ensure cached TSV is a Tensor with matching shape
                if isinstance(cached_tsv, torch.Tensor):
                    # Expected TSV shape: [num_subgraphs, 6]
                    if cached_tsv.dim() == 2 and cached_tsv.shape[0] == num_subgraphs:
                        logger.info(
                            f"Loaded TSV from cache: shape={tuple(cached_tsv.shape)} matches num_subgraphs={num_subgraphs}"
                        )
                        return cached_tsv.to(device)
                    else:
                        logger.warning(
                            f"Cached TSV shape mismatch: shape={tuple(cached_tsv.shape)}, expected first dim={num_subgraphs}; recomputing"
                        )
                else:
                    logger.warning(
                        f"Cached TSV has invalid type: {type(cached_tsv)}; recomputing"
                    )
            except Exception as e:
                logger.warning(f"Failed to load TSV cache: {e}; recomputing")

        # Compute PageRank, clustering coefficient, and degree on the private graph
        logger.info("Computing PageRank, clustering coefficient, and degree on the private graph...")

        # 1) Degree
        deg = self._degree_vector(private_data).float().to(device)

        # 2) PageRank and clustering (via cuGraph)
        pagerank = torch.zeros(
            private_data.num_nodes, dtype=torch.float32, device=device
        )
        clustering = torch.zeros(
            private_data.num_nodes, dtype=torch.float32, device=device
        )

        try:
            # Convert to cuGraph format
            edges = private_data.edge_index.cpu().numpy()
            df = cudf.DataFrame()
            df["src"] = edges[0].astype(np.int32)
            df["dst"] = edges[1].astype(np.int32)

            # Create cuGraph graph
            G_cu = cugraph.Graph()
            G_cu.from_cudf_edgelist(df, source="src", destination="dst")

            # PageRank
            try:
                pagerank_df = cugraph.pagerank(G_cu)
                pagerank_pandas = pagerank_df.to_pandas()
                if len(pagerank_pandas) > 0:
                    pagerank_values = (
                        torch.from_numpy(pagerank_pandas["pagerank"].values)
                        .float()
                        .to(device)
                    )
                    pagerank_indices = (
                        torch.from_numpy(pagerank_pandas["vertex"].values)
                        .long()
                        .to(device)
                    )
                    # Ensure indices are in bounds
                    valid_mask = pagerank_indices < private_data.num_nodes
                    if valid_mask.any():
                        pagerank[pagerank_indices[valid_mask]] = pagerank_values[
                            valid_mask
                        ]
            except Exception as e:
                logger.warning(f"Error computing PageRank: {e}")

            # Clustering coefficient
            try:
                triangle_counts = cugraph.triangle_count(G_cu)
                triangle_counts_df = triangle_counts.to_pandas()
                degrees = G_cu.degree()
                degrees_df = degrees.to_pandas()

                if len(triangle_counts_df) > 0 and len(degrees_df) > 0:
                    # Merge triangle counts and degree information
                    merged_df = triangle_counts_df.merge(
                        degrees_df, left_on="vertex", right_on="vertex", how="outer"
                    )
                    merged_df = merged_df.fillna(0)

                    # Clustering coefficient: 2 * triangle_count / (degree * (degree-1))
                    merged_df["clustering_coeff"] = merged_df.apply(
                        lambda row: 2
                        * row["counts"]
                        / (row["degree"] * (row["degree"] - 1))
                        if row["degree"] > 1
                        else 0.0,
                        axis=1,
                    )

                    # Fill clustering coefficients
                    for _, row in merged_df.iterrows():
                        node = int(row["vertex"])
                        if node < private_data.num_nodes:
                            clustering[node] = float(row["clustering_coeff"])
            except Exception as e:
                logger.warning(f"Error computing clustering coefficient: {e}")
        except Exception as e:
            logger.warning(f"Error computing graph metrics with cuGraph: {e}; falling back to default zeros")

        # 3) TSV per subgraph
        tsv_list = []
        for sg in subgraphs:
            # Original node indices
            if hasattr(sg, "original_node_indices"):
                original_indices = sg.original_node_indices.to(device)
            else:
                # Backward-compat: if original indices are missing, use local subgraph indices
                original_indices = torch.arange(sg.num_nodes, device=device)

            if original_indices.numel() == 0:
                tsv_list.append(torch.zeros(6, device=device))
                continue

            # Extract features and labels from the full graph
            x = private_data.x[original_indices].to(device)
            y = private_data.y[original_indices].float().to(device)

            # Subgraph metrics
            I_fra = y.mean() if y.numel() > 0 else torch.tensor(0.0, device=device)
            I_fea = torch.norm(x, dim=1).mean()
            I_deg = (
                deg[original_indices].mean()
                if original_indices.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            I_clu = (
                clustering[original_indices].mean()
                if original_indices.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            I_var = torch.var(x, dim=0).mean()
            I_pr = (
                pagerank[original_indices].mean()
                if original_indices.numel() > 0
                else torch.tensor(0.0, device=device)
            )

            tsv_list.append(torch.stack([I_fra, I_fea, I_deg, I_clu, I_var, I_pr]))

        tsv = torch.stack(tsv_list, dim=0).to(torch.float32)

        # Save cache
        if cache_path is not None:
            os.makedirs(
                os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
                exist_ok=True,
            )
            try:
                torch.save(tsv, cache_path)
                logger.info(f"TSV saved to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save TSV cache: {e}")

        return tsv.to(device)
