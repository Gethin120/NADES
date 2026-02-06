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

# 确保 PyTorch 和 cuGraph/cuDF 在同一设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CugraphOverlapPartitioner:
    """
    使用 cugraph 加速，实现多种受控重叠图划分策略。

    方法 A: Louvain/Spectral + 膨胀 (Expansion)
    方法 B: Louvain/1-Hop + 剪枝 (Pruning)
    方法 C: Louvain 社区检测 + 直接构建子图 (Community Detection)
    方法 D: CAOP (Controlled Overlap Partitioning) - 基于特征和度相似性的划分
    方法 E: 基于方法 B 的模合并剪枝 (先合并自然社区到 S_hint 个宏社区，再执行剪枝)
    方法 F: Louvain + 宏社区 + 核心优先剪枝 (Louvain社区发现 -> 过滤 -> 模合并 -> 1-hop扩展 -> 核心优先剪枝)
    方法 G: 全局最优剪枝 (两阶段：生成多样化候选子图池 -> 全局max_overlap=3剪枝)
    方法 H: 标签引导 CAOP (在方法D基础上，种子由已知/未知标签节点混合构成，并保证每个社区中心包含已知标签)
    方法 I: 标签引导 CAOP (变体)
    方法 J: 标签均衡全局剪枝 (在方法G基础上，剔除不同时包含 y=0/1 的候选子图)
    """

    def __init__(
        self,
        device=DEVICE,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
    ):
        """
        初始化划分器。

        参数:
            device: 计算设备
            train_ratio: 训练集比例（所有方法共用）
            val_ratio: 验证集比例（所有方法共用）
        """
        self.device = device
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.min_community_size = 100
        if not torch.cuda.is_available():
            raise EnvironmentError("cugraph 需要 CUDA 环境")

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
        统一的划分方法接口，根据 method 参数选择不同的划分策略。

        参数:
            method: 划分方法，可选 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'
                - 'A': Expansion (膨胀法)
                - 'B': Pruning (剪枝法)
                - 'C': Community Detection (社区检测法)
                - 'D': CAOP (受控重叠划分法)
                - 'E': 模合并剪枝法（宏社区）
                - 'F': Louvain + 宏社区 + 核心优先剪枝法
                - 'G': 全局最优剪枝法（两阶段方法）
                - 'H': 标签引导 CAOP
                - 'I': 标签引导 CAOP (修改版)
                - 'J': 标签均衡全局剪枝
            data: 输入图数据
            S: 目标分区数（方法 A、C、D、E、F 使用，方法 C 和 F 作为宏社区数量）
            s_max: 每个节点的最大重叠社区数（方法 A、B、D 使用，方法 F 固定为 1）
            resolution: Louvain 算法的 resolution 参数（方法 C 使用）
            cache_path: 缓存文件路径
            logger: 日志记录器

        返回:
            子图列表
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
        elif method == "E":
            return self.partition_modular_macro(data, S, s_max, cache_path, logger)
        elif method == "F":
            return self.partition_core_edge(
                data,
                S_F=S,
                cache_path=cache_path,
                logger=logger,
            )
        elif method == "G":
            # 方法 G: 全局最优剪枝，使用 S 作为 N_D 和 N_E 的提示值
            N_D = 20
            N_E = 20
            return self.partition_global_optimal(
                data,
                N_D=N_D,
                N_E=N_E,
                cache_path=cache_path,
                logger=logger,
            )
        elif method == "H":
            return self.partition_label_guided_caop(
                data,
                S=S,
                s_max=s_max,
                cache_path=cache_path,
                logger=logger,
            )
        elif method == "I":
            return self.partition_label_guided_caop_modified(
                data=data,
                S=S,
                s_max=s_max,
                cache_path=cache_path,
                logger=logger,
            )
        elif method == "J":
            N_D = 20
            N_E = 20
            return self.partition_label_balanced_global(
                data,
                N_D=N_D,
                N_E=N_E,
                cache_path=cache_path,
                logger=logger,
            )
        else:
            raise ValueError(
                f"未知的划分方法 '{method}'，可选: 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'"
            )

    def _pyg_to_cugraph_graph(self, data: Data) -> cugraph.Graph:
        """
        辅助函数：将 PyG Data 对象转换为 cugraph.Graph。
        假设 data.edge_index 是无向的，且节点已从 0..N-1 编号。
        确保顶点类型为 int32，以满足 spectralBalancedCutClustering 的要求。
        """
        # 1. 将 PyG 的 edge_index (Tensor) 转换为 cuDF DataFrame
        #    cugraph 需要 [src, dst] 格式，且类型为 int32
        edge_index_cpu = data.edge_index.cpu().numpy()
        gdf = cudf.DataFrame()
        # 确保类型为 int32（spectralBalancedCutClustering 的要求）
        gdf["src"] = edge_index_cpu[0].astype(np.int32)
        gdf["dst"] = edge_index_cpu[1].astype(np.int32)

        # 2. 从 cuDF Edgelist 创建 cugraph.Graph
        # renumber=False 假设 PyG 节点索引已经是 0...N-1
        G_cu = cugraph.Graph(directed=False)
        G_cu.from_cudf_edgelist(gdf, source="src", destination="dst", renumber=False)
        return G_cu

    def _load_cache(
        self, cache_path: Optional[str], logger=None
    ) -> Optional[List[Data]]:
        """
        统一的缓存加载函数。

        参数:
            cache_path: 缓存文件路径
            logger: 日志记录器（可选）

        返回:
            如果缓存存在且有效，返回缓存的子图列表；否则返回 None
        """
        if cache_path is None or not os.path.exists(cache_path):
            return None

        try:
            cached = torch.load(cache_path)
            if isinstance(cached, list):
                logger.info(f"从缓存加载子图: {len(cached)} 个")
                return cached
            logger.warning(
                f"缓存文件 {cache_path} 类型为 {type(cached)}，重新构建子图。"
            )
        except Exception as e:
            logger.warning(f"加载缓存失败：{e}，正在重新计算。")

        return None

    def _save_cache(
        self, subgraphs: List[Data], cache_path: Optional[str], logger=None
    ) -> None:
        """
        统一的缓存保存函数。

        参数:
            subgraphs: 要保存的子图列表
            cache_path: 缓存文件路径
            logger: 日志记录器（可选）
        """
        if cache_path is None:
            return

        try:
            os.makedirs(
                os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
                exist_ok=True,
            )
            torch.save(subgraphs, cache_path)
            logger.info(f"子图已保存到缓存: {cache_path}")
        except Exception as e:
            logger.warning(f"保存缓存失败：{e}")

    def _build_pyg_subgraphs(
        self,
        original_data: Data,
        node_to_community_map: List[Set[int]],
        S: int,
    ) -> List[Data]:
        """
        辅助函数：根据最终的节点->社区映射，构建 PyG Data 对象列表。
        """

        # 1. 反转映射：从 (节点 -> 社区) 变为 (社区 -> 节点)
        # final_partitions[i] = [所有属于社区 i 的节点 ID 列表]
        final_partitions: List[List[int]] = [[] for _ in range(S)]
        for node_id, community_set in enumerate(node_to_community_map):
            for community_id in community_set:
                if community_id < S:  # 确保不越界
                    final_partitions[community_id].append(node_id)

        # 2. 为每个分区创建 PyG Data 对象
        subgraph_list: List[Data] = []
        # 获取原始数据的设备，确保所有张量在同一设备上
        device = original_data.edge_index.device

        for community_id in range(S):
            original_node_indices = torch.tensor(
                final_partitions[community_id], dtype=torch.long, device=device
            )
            if original_node_indices.numel() == 0:
                continue  # 跳过空社区

            # 3. 使用 PyG 的 subgraph 工具提取边，并重新编号节点
            # relabel_nodes=True 会将 [10, 20, 30] 映射到 [0, 1, 2]
            # 确保 edge_index 和 node_indices 在同一设备上
            edge_index = original_data.edge_index.to(device)
            sub_edge_index, _ = pyg_subgraph(
                original_node_indices,
                edge_index,
                relabel_nodes=True,
                num_nodes=original_data.num_nodes,
            )

            # 4. 提取节点特征和标签
            sub_x = original_data.x[original_node_indices]
            sub_y = original_data.y[original_node_indices]

            # 5. 创建新的 Data 对象
            num_sub_nodes = sub_x.size(0)
            sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)
            sub_data.num_nodes = num_sub_nodes
            # 保存原始节点索引，用于后续计算 TSV
            sub_data.original_node_indices = original_node_indices

            # 6. 使用 RandomNodeSplit 创建训练/验证/测试掩码
            # 计算测试集比例（剩余部分）
            test_ratio = 1.0 - self.train_ratio - self.val_ratio
            splitter = RandomNodeSplit(
                split="train_rest",  # train 是剩余部分
                num_val=self.val_ratio,  # 验证集比例
                num_test=test_ratio,  # 测试集比例
                key="y",  # 基于标签进行分层划分
            )
            sub_data = splitter(sub_data)

            # 确保 test_mask 存在（如果 test_ratio 为 0，则创建全 False 的掩码）
            if not hasattr(sub_data, "test_mask") or sub_data.test_mask is None:
                sub_data.test_mask = torch.zeros(
                    num_sub_nodes, dtype=torch.bool, device=sub_data.x.device
                )

            # 验证子图有效性：至少2个节点、有边、有训练节点、训练集有多个类别
            if sub_data.num_nodes < 2:
                continue  # 跳过节点数过少的子图

            if sub_data.edge_index.size(1) == 0:
                continue  # 跳过没有边的子图

            if not hasattr(sub_data, "train_mask") or not sub_data.train_mask.any():
                continue  # 跳过没有训练节点的子图

            # 检查训练集中是否有多个类别（避免单一类别导致权重计算问题）
            train_labels = sub_data.y[sub_data.train_mask]
            unique_labels = torch.unique(train_labels)
            if len(unique_labels) < 2:
                continue  # 跳过训练集只有单一类别的子图
            # 检查验证集是否有多个类别
            val_labels = sub_data.y[sub_data.val_mask]
            unique_labels = torch.unique(val_labels)
            if len(unique_labels) < 2:
                continue  # 跳过验证集只有单一类别的子图

            subgraph_list.append(sub_data)

        return subgraph_list

    # --- 方法 A: cugraph.spectralBalancedCut + 膨胀 ---

    def partition_expansion(
        self,
        data: Data,
        S: int,
        s_max: int,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 A (膨胀法):
        1. 使用 cugraph.spectralBalancedCut 进行 s_max=1 的硬划分（初始化）。
        2. 基于特征亲和力，将节点"膨胀"到其他社区，直到 s_max。

        参数:
            data: 输入图数据
            S: 目标分区数
            s_max: 每个节点的最大重叠社区数
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器（可选）
        """
        # 检查缓存
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"开始执行 [方法 A：膨胀法] (S={S}, s_max={s_max})")
        num_nodes = data.num_nodes

        # 1. cugraph 加速的硬划分 (初始化)
        # 我们使用 spectralBalancedCut 来确保得到 S 个分区
        G_cu = self._pyg_to_cugraph_graph(data)
        logger.info("... cugraph.spectralBalancedCut 运行中 ...")
        partition_df = cugraph.spectralBalancedCutClustering(G_cu, num_clusters=S)

        # 2. 构建核心社区和计算质心
        # 检查返回的 DataFrame 结构
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

        # 3. 膨胀逻辑 (CAOP 的第二步)
        logger.info(f"... 膨胀 {num_nodes} 个节点至 s_max={s_max} ...")
        node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]
        node_counts = np.zeros(num_nodes, dtype=int)

        # 3a. 初始化：分配核心社区
        for part_id, nodes in enumerate(core_communities):
            for node in nodes:
                node_to_community_map[node].add(part_id)
                node_counts[node] = 1

        # 3b. 贪婪膨胀：为每个节点找到亲和力最高的 s_max 个社区
        for node in range(num_nodes):
            if node_counts[node] >= s_max:
                continue

            node_feature = x_cpu[node].reshape(1, -1)

            # 计算与所有S个质心的亲和力
            affinities = cosine_similarity(node_feature, centroids)[0]

            # 降序排列社区 ID
            sorted_community_ids = np.argsort(affinities)[::-1]

            # 添加到亲和力最高的社区，直到 s_max
            for community_id in sorted_community_ids:
                if node_counts[node] >= s_max:
                    break
                # add() 自动处理重复
                if community_id not in node_to_community_map[node]:
                    node_to_community_map[node].add(community_id)
                    node_counts[node] += 1

        logger.info("... 膨胀完成，正在构建 PyG 子图 ...")
        subgraphs = self._build_pyg_subgraphs(data, node_to_community_map, S)

        # 保存缓存
        self._save_cache(subgraphs, cache_path, logger)

        return subgraphs

    # --- 方法 B: cugraph.louvain + 1跳邻居 + 剪枝 ---

    def partition_pruning(
        self,
        data: Data,
        S_hint: int,
        s_max: int,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 B (剪枝法):
        1. 使用 cugraph.louvain 找到 S_prime 个"自然"社区。
        2. 通过 1-hop 邻居创建"不受控"的重叠（模拟 SLLP）。
        3. 基于特征亲和力，"剪枝"多余的社区，直到 s_max。

        注意: S_hint 在此方法中被忽略，最终社区数 S 由 Louvain 决定。

        参数:
            data: 输入图数据
            S_hint: 提示的社区数量（实际会被忽略）
            s_max: 每个节点的最大重叠社区数
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器（可选）
        """
        # 检查缓存
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"开始执行 [方法 B：剪枝法] (s_max={s_max})")
        num_nodes = data.num_nodes

        # 1. cugraph 加速的硬划分 (Louvain)
        G_cu = self._pyg_to_cugraph_graph(data)
        logger.info("... cugraph.louvain 运行中 ...")
        partition_df = cugraph.louvain(G_cu)
        partition_df = partition_df[0]
        partition_pd = partition_df.to_pandas()

        partition_map = {row.vertex: row.partition for row in partition_pd.itertuples()}
        S_found = partition_pd["partition"].max() + 1
        logger.info(f"... Louvain 找到了 {S_found} 个自然社区 ...")

        # 2. 创建 1-Hop 邻居重叠（模拟 SLLP 的不受控重叠）
        logger.info("... 创建 1-Hop 邻居重叠 ...")
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
            # 添加节点自己的社区
            if node in partition_map:
                initial_node_to_community_map[node].add(partition_map[node])
            # 添加所有 1-hop 邻居的社区
            for neighbor in adj.get(node, set()):
                if neighbor in partition_map:
                    initial_node_to_community_map[node].add(partition_map[neighbor])

            if len(initial_node_to_community_map[node]) > max_overlap_found:
                max_overlap_found = len(initial_node_to_community_map[node])

        logger.info(f"... 发现的最大不受控重叠 s = {max_overlap_found} ...")

        # 3. 剪枝逻辑
        logger.info(f"... 剪枝 {num_nodes} 个节点至 s_max={s_max} ...")
        x_cpu = data.x.cpu().numpy()

        # 3a. 计算所有 S_found 个社区的质心
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

        # 3b. 剪枝
        final_node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]

        for node in range(num_nodes):
            communities = list(initial_node_to_community_map[node])

            if len(communities) == 0:
                continue
            elif len(communities) <= s_max:
                final_node_to_community_map[node] = set(communities)
            else:
                # 亲和力计算和剪枝
                node_feature = x_cpu[node].reshape(1, -1)

                # 只计算节点所在社区的质心
                community_centroids = centroids[communities]

                # 计算与这些社区的亲和力
                affinities = cosine_similarity(node_feature, community_centroids)[0]

                # 降序排列 *本地* 索引
                sorted_local_indices = np.argsort(affinities)[::-1]

                # 保留亲和力最高的 s_max 个
                for i in range(s_max):
                    local_idx = sorted_local_indices[i]
                    global_community_id = communities[local_idx]
                    final_node_to_community_map[node].add(global_community_id)

        logger.info("... 剪枝完成，正在构建 PyG 子图 ...")
        subgraphs = self._build_pyg_subgraphs(
            data, final_node_to_community_map, S_found
        )

        # 过滤掉节点数量小于 min_community_size 的子图
        filtered_subgraphs = [
            sg for sg in subgraphs if sg.num_nodes >= self.min_community_size
        ]
        num_filtered = len(subgraphs) - len(filtered_subgraphs)

        if num_filtered > 0:
            logger.info(
                f"过滤掉 {num_filtered} 个节点数量小于 {self.min_community_size} 的子图"
            )

        if len(filtered_subgraphs) == 0:
            error_msg = (
                f"过滤后没有剩余子图（所有子图都小于 {self.min_community_size} 个节点）"
            )
            logger.error(error_msg)
            return []

        # 保存缓存
        self._save_cache(filtered_subgraphs, cache_path, logger)

        return filtered_subgraphs

    # --- 方法 C: cugraph.louvain + 直接构建子图 ---

    def partition_community_detection(
        self,
        data: Data,
        resolution: float = 1.0,
        S_hint: Optional[int] = None,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 C (社区检测法):
        1. 使用 cugraph.louvain 进行社区检测（支持 resolution 参数）。
        2. 直接为每个社区构建子图，不进行膨胀或剪枝。
        3. 支持磁盘缓存（可选）。

        参数:
            data: 输入图数据
            resolution: Louvain 算法的 resolution 参数，控制社区大小
            S_hint: 如果指定，则使用模运算将自然社区合并为 S_hint 个宏社区
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器（可选）

        返回:
            子图列表
        """
        # 检查缓存
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"开始执行 [方法 C：社区检测法] (resolution={resolution})")
        num_nodes = data.num_nodes

        # 1. 使用 cugraph.louvain 进行社区检测
        G_cu = self._pyg_to_cugraph_graph(data)
        logger.info("... cugraph.louvain 运行中 ...")
        communities: Dict[int, List[int]] = {}

        try:
            # 尝试使用 resolution 参数
            try:
                parts_df, _ = cugraph.louvain(G_cu, resolution=resolution)
            except (TypeError, ValueError):
                # 如果不支持 resolution 参数，使用默认调用
                logger.info("cugraph.louvain 不支持 resolution 参数，使用默认调用")
                parts_df = cugraph.louvain(G_cu)

            # 转换为 pandas
            if isinstance(parts_df, tuple):
                parts_df = parts_df[0]
            parts_pd = parts_df.to_pandas()

            for _, row in parts_pd.iterrows():
                node_id = int(row["vertex"])
                partition_id = int(row["partition"])
                communities.setdefault(partition_id, []).append(node_id)

        except Exception as e:
            logger.warning(f"Louvain 社区检测失败（{e}），将退化为单节点社区。")

        if not communities:
            logger.warning("社区检测没有产生有效划分，将使用单节点社区作为回退方案。")
            all_nodes = torch.arange(num_nodes, device=self.device)
            communities = {
                idx: [int(node)] for idx, node in enumerate(all_nodes.cpu().tolist())
            }

        num_communities = len(communities)
        logger.info(f"... Louvain 检测到 {num_communities} 个社区 ...")

        # 输出每个社区的节点数量
        community_sizes = {
            comm_id: len(nodes) for comm_id, nodes in communities.items()
        }
        # sorted_comm_ids = sorted(community_sizes.keys())
        # logger.info("... 各社区节点数量统计:")
        # for comm_id in sorted_comm_ids:
        #     size = community_sizes[comm_id]
        #     logger.info(f"  社区 {comm_id}: {size} 个节点")

        # 输出统计信息
        sizes_list = list(community_sizes.values())
        if sizes_list:
            min_size = min(sizes_list)
            max_size = max(sizes_list)
            avg_size = sum(sizes_list) / len(sizes_list)
            logger.info(
                f"  社区大小统计: 最小={min_size}, 最大={max_size}, 平均={avg_size:.1f}"
            )

        # 过滤掉节点数量小于 min_community_size 的社区
        filtered_communities = {
            comm_id: nodes
            for comm_id, nodes in communities.items()
            if len(nodes) >= self.min_community_size
        }
        num_filtered = len(communities) - len(filtered_communities)

        if num_filtered > 0:
            logger.info(
                f"过滤掉 {num_filtered} 个节点数量小于 {self.min_community_size} 的社区"
            )
            # 输出被过滤的社区信息
            filtered_sizes = [
                (comm_id, len(nodes))
                for comm_id, nodes in communities.items()
                if len(nodes) < self.min_community_size
            ]
            if filtered_sizes:
                filtered_info = ", ".join(
                    [
                        f"社区{comm_id}({size}节点)"
                        for comm_id, size in filtered_sizes[:10]
                    ]
                )
                if len(filtered_sizes) > 10:
                    filtered_info += f" ... (共{len(filtered_sizes)}个)"
                logger.info(f"  被过滤的社区: {filtered_info}")

        communities = filtered_communities
        num_communities = len(communities)

        # 若指定 S_hint，则在过滤后进行宏社区合并
        if S_hint is not None:
            if S_hint <= 0:
                raise ValueError("S_hint 必须为正整数")
            logger.info(f"... 合并社区为 {S_hint} 个宏社区 (模运算) ...")
            macro_communities = {i: [] for i in range(S_hint)}
            for comm_id, nodes in communities.items():
                macro_id = comm_id % S_hint
                macro_communities[macro_id].extend(nodes)
            communities = {k: v for k, v in macro_communities.items() if len(v) > 0}
            num_communities = len(communities)
            logger.info(
                "宏社区节点数量: "
                + ", ".join([f"Macro {k}: {len(v)}" for k, v in communities.items()])
            )

        if num_communities == 0:
            error_msg = (
                f"过滤后没有剩余社区（所有社区都小于 {self.min_community_size} 个节点）"
            )
            logger.error(error_msg)
            # 返回空列表或使用所有节点作为单个社区
            return []

        logger.info(f"过滤后剩余 {num_communities} 个社区")

        # 2. 将社区映射转换为节点到社区的映射（每个节点只属于一个社区）
        node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]
        for comm_id, nodes in communities.items():
            for node_id in nodes:
                if node_id < num_nodes:
                    node_to_community_map[node_id].add(comm_id)

        # 3. 使用 _build_pyg_subgraphs 构建子图
        logger.info("... 使用 _build_pyg_subgraphs 构建子图 ...")
        subgraphs = self._build_pyg_subgraphs(
            data, node_to_community_map, num_communities
        )

        logger.info(f"... 构建完成，共 {len(subgraphs)} 个子图 ...")

        # 4. 保存缓存
        self._save_cache(subgraphs, cache_path, logger)

        return subgraphs

    # --- 方法 D: CAOP (Controlled Overlap Partitioning) ---

    def _degree_vector(self, data: Data) -> torch.Tensor:
        """计算节点的度向量"""
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
        "核心优先"剪枝：在 s_max=1 约束下，优先保留宏社区的核心节点。

        参数:
            macro_communities: 宏社区列表，每个元素是核心节点ID的集合
            candidate_subgraphs: 候选子图列表（宏社区节点 + 1-hop邻居）
            data: 输入图数据
            logger: 日志记录器（可选）

        返回:
            剪枝后的子图列表
        """
        x_cpu = data.x.cpu().numpy()
        num_subgraphs = len(candidate_subgraphs)

        # 计算每个子图的质心（用于相似度计算）
        subgraph_centroids = []
        for sg_nodes in candidate_subgraphs:
            if len(sg_nodes) > 0:
                nodes_list = list(sg_nodes)
                centroid = x_cpu[nodes_list].mean(axis=0)
                subgraph_centroids.append(centroid)
            else:
                subgraph_centroids.append(np.zeros(x_cpu.shape[1]))

        # 记录每个节点被分配到的子图
        node_to_subgraph: Dict[int, int] = {}
        final_subgraphs: List[Set[int]] = [set() for _ in range(num_subgraphs)]

        # 第一遍：处理核心节点（优先分配）
        for subgraph_id in range(num_subgraphs):
            core_nodes = macro_communities[subgraph_id]
            for node_id in core_nodes:
                if node_id not in node_to_subgraph:
                    # 核心节点尚未分配，直接分配给其宏社区
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                else:
                    # 核心节点已分配给其他子图，需要仲裁
                    old_subgraph_id = node_to_subgraph[node_id]
                    node_feature = x_cpu[node_id].reshape(1, -1)

                    # 计算与两个子图的相似度
                    old_centroid = subgraph_centroids[old_subgraph_id].reshape(1, -1)
                    new_centroid = subgraph_centroids[subgraph_id].reshape(1, -1)

                    old_sim = cosine_similarity(node_feature, old_centroid)[0, 0]
                    new_sim = cosine_similarity(node_feature, new_centroid)[0, 0]

                    # 核心节点优先：如果新子图更相似，则重新分配
                    if new_sim > old_sim:
                        final_subgraphs[old_subgraph_id].remove(node_id)
                        node_to_subgraph[node_id] = subgraph_id
                        final_subgraphs[subgraph_id].add(node_id)

        # 第二遍：处理非核心节点（1-hop邻居）
        for subgraph_id in range(num_subgraphs):
            core_nodes = macro_communities[subgraph_id]
            candidate_nodes = candidate_subgraphs[subgraph_id]
            non_core_nodes = candidate_nodes - core_nodes

            for node_id in non_core_nodes:
                if node_id not in node_to_subgraph:
                    # 非核心节点尚未分配，可以加入
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                # 如果非核心节点已经在其他子图中，则不加入（严格执行s_max=1）

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
        "核心优先+欺诈优先"剪枝：在 s_max=1 约束下，优先保留宏社区的核心节点，
        并考虑欺诈标签的优先级。

        优先级：
        1. 最高优先级：核心欺诈节点（属于宏社区且是欺诈节点）
        2. 次高优先级：非核心欺诈节点（1-hop邻居中的欺诈节点）
        3. 最低优先级：正常节点

        参数:
            macro_communities: 宏社区列表，每个元素是核心节点ID的集合
            candidate_subgraphs: 候选子图列表（宏社区节点 + 1-hop邻居）
            data: 输入图数据（必须包含 y 标签）
            fraud_weight: 欺诈节点的加权分数（用于调整相似度）
            logger: 日志记录器（可选）

        返回:
            剪枝后的子图列表
        """
        x_cpu = data.x.cpu().numpy()
        num_subgraphs = len(candidate_subgraphs)

        # 获取欺诈标签（假设欺诈标签为1，正常为0）
        if not hasattr(data, "y") or data.y is None:
            logger.warning("数据中没有标签，回退到标准核心优先剪枝")
            return self._core_priority_pruning(
                macro_communities, candidate_subgraphs, data, logger
            )

        y_cpu = data.y.cpu().numpy()
        # 假设二分类：1为欺诈，0为正常（如果是多分类，需要调整）
        if y_cpu.dtype in [np.int64, np.int32]:
            fraud_labels = (y_cpu == 1).astype(np.float32)
        else:
            # 如果是浮点数，假设 > 0.5 为欺诈
            fraud_labels = (y_cpu > 0.5).astype(np.float32)

        # 计算每个子图的质心（用于相似度计算）
        subgraph_centroids = []
        for sg_nodes in candidate_subgraphs:
            if len(sg_nodes) > 0:
                nodes_list = list(sg_nodes)
                centroid = x_cpu[nodes_list].mean(axis=0)
                subgraph_centroids.append(centroid)
            else:
                subgraph_centroids.append(np.zeros(x_cpu.shape[1]))

        # 记录每个节点被分配到的子图
        node_to_subgraph: Dict[int, int] = {}
        final_subgraphs: List[Set[int]] = [set() for _ in range(num_subgraphs)]

        # 第一遍：处理核心节点（按欺诈优先级）
        for subgraph_id in range(num_subgraphs):
            core_nodes = macro_communities[subgraph_id]
            # 分离核心欺诈节点和核心正常节点
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

            # 优先处理核心欺诈节点
            for node_id in core_fraud_nodes:
                if node_id not in node_to_subgraph:
                    # 核心欺诈节点尚未分配，直接分配给其宏社区
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                else:
                    # 核心欺诈节点已分配给其他子图，需要仲裁
                    old_subgraph_id = node_to_subgraph[node_id]
                    node_feature = x_cpu[node_id].reshape(1, -1)

                    # 计算与两个子图的相似度
                    old_centroid = subgraph_centroids[old_subgraph_id].reshape(1, -1)
                    new_centroid = subgraph_centroids[subgraph_id].reshape(1, -1)

                    old_sim = cosine_similarity(node_feature, old_centroid)[0, 0]
                    new_sim = cosine_similarity(node_feature, new_centroid)[0, 0]

                    # 核心欺诈节点优先：如果新子图更相似，则重新分配
                    if new_sim > old_sim:
                        final_subgraphs[old_subgraph_id].remove(node_id)
                        node_to_subgraph[node_id] = subgraph_id
                        final_subgraphs[subgraph_id].add(node_id)

            # 处理核心正常节点
            for node_id in core_normal_nodes:
                if node_id not in node_to_subgraph:
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                else:
                    # 核心正常节点已分配，根据相似度仲裁
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

        # 第二遍：处理非核心节点（1-hop邻居），按欺诈优先级
        for subgraph_id in range(num_subgraphs):
            core_nodes = macro_communities[subgraph_id]
            candidate_nodes = candidate_subgraphs[subgraph_id]
            non_core_nodes = candidate_nodes - core_nodes

            # 分离非核心欺诈节点和正常节点
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

            # 优先处理非核心欺诈节点
            for node_id in non_core_fraud_nodes:
                if node_id not in node_to_subgraph:
                    # 非核心欺诈节点尚未分配，可以加入
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                else:
                    # 非核心欺诈节点已在其他子图中，需要根据加权相似度仲裁
                    old_subgraph_id = node_to_subgraph[node_id]
                    node_feature = x_cpu[node_id].reshape(1, -1)

                    old_centroid = subgraph_centroids[old_subgraph_id].reshape(1, -1)
                    new_centroid = subgraph_centroids[subgraph_id].reshape(1, -1)

                    old_sim = cosine_similarity(node_feature, old_centroid)[0, 0]
                    new_sim = cosine_similarity(node_feature, new_centroid)[0, 0]

                    # 给欺诈节点加权：检查两个子图中哪个包含更多欺诈节点
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

                    # 加权相似度：如果新子图包含更多欺诈节点，给予额外加分
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

            # 处理非核心正常节点（最低优先级）
            for node_id in non_core_normal_nodes:
                if node_id not in node_to_subgraph:
                    # 非核心正常节点尚未分配，可以加入
                    node_to_subgraph[node_id] = subgraph_id
                    final_subgraphs[subgraph_id].add(node_id)
                # 如果非核心正常节点已经在其他子图中，则不加入（严格执行s_max=1）

        return final_subgraphs

    # --- 方法 D: CAOP法 ---

    def partition_caop(
        self,
        data: Data,
        S: int,
        s_max: int,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 D (CAOP法):
        基于特征相似性和度相似性的受控重叠图划分算法。

        参数:
            data: 输入图数据
            S: 目标分区数
            s_max: 每个节点的最大重叠社区数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器（可选）

        返回:
            子图列表
        """
        # 检查缓存
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"开始执行 [方法 D：CAOP法] (S={S}, s_max={s_max})")
        device = self.device
        x = data.x.to(device)
        n = data.num_nodes
        deg = self._degree_vector(data).float().to(device)
        max_deg = deg.max().clamp_min(1.0)
        x = F.normalize(x, dim=1)

        # 1. 选择种子节点
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

        # 2. 初始化分组
        groups: List[List[int]] = [[s] for s in seeds]
        assigned_counts = torch.zeros(n, dtype=torch.long, device=device)
        assigned_counts[torch.tensor(seeds, device=device)] = 1
        group_feat_sum = x[torch.tensor(seeds, device=device)].clone()
        group_deg_sum = deg[torch.tensor(seeds, device=device)].clone()
        group_counts = torch.ones(S, device=device)

        # 3. 贪婪分配节点到社区
        K = min(8, S)
        batch_size = 20000 if n > 50000 else 10000
        logger.info(f"... 分配 {n} 个节点到 {S} 个社区 (s_max={s_max}) ...")

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

        # 4. 生成子图
        logger.info("... 生成子图 ...")
        subgraphs = []
        edge_device = data.edge_index.device

        for nodes in groups:
            if not nodes:
                continue

            # 转换为排序后的节点索引张量
            node_indices = torch.tensor(
                sorted(set(int(i) for i in nodes)), dtype=torch.long, device=edge_device
            )

            # 使用 PyG 的 subgraph 工具提取子图
            sub_edge_index, edge_mask = pyg_subgraph(
                node_indices,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )

            # 创建子图数据对象
            d = Data(
                x=data.x[node_indices],
                edge_index=sub_edge_index,
                y=data.y[node_indices],
            )
            d.num_nodes = node_indices.numel()
            # 保存原始节点索引，用于后续计算 TSV
            d.original_node_indices = node_indices

            # 使用 RandomNodeSplit 创建训练/验证/测试掩码
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
        )  # yelp和amazon不需要，elliptic需要。
        logger.info(f"... 构建完成，共 {len(subgraphs)} 个子图 ...")

        # 5. 保存缓存
        self._save_cache(subgraphs, cache_path, logger)

        return subgraphs

    # --- 方法 H: 标签引导的 CAOP (已知/未知混合种子) ---

    def partition_label_guided_caop(
        self,
        data: Data,
        S: int,
        s_max: int,
        labeled_seed_ratio: float = 0.6,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 H (标签引导 CAOP):
        在 CAOP 基础上，种子节点由已知标签 (y=0/1) 与未知标签 (y=-1) 的节点混合组成，
        并为未知标签种子配对一个最近的已知标签锚点，确保每个初始微型社区中心拥有明确标签。
        """
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        device = self.device
        x = data.x.to(device)
        n = data.num_nodes
        deg = self._degree_vector(data).float().to(device)
        max_deg = deg.max().clamp_min(1.0)
        x = F.normalize(x, dim=1)

        y = data.y.view(-1).cpu()
        known_mask = (y == 0) | (y == 1)
        unknown_mask = y < 0
        labeled_idx = torch.nonzero(known_mask, as_tuple=False).view(-1)
        unlabeled_idx = torch.nonzero(unknown_mask, as_tuple=False).view(-1)

        if labeled_idx.numel() == 0:
            logger.warning("方法 H: 未找到已知标签节点，回退至方法 D。")
            return self.partition_caop(
                data, S=S, s_max=s_max, cache_path=cache_path, logger=logger
            )

        rng = torch.Generator(device="cpu")

        def _farthest_sampling(pool: torch.Tensor, num: int) -> List[int]:
            if num <= 0 or pool.numel() == 0:
                return []
            pool = pool.to(device)
            num = min(num, pool.numel())
            first_pos = torch.randint(
                low=0, high=pool.numel(), size=(1,), generator=rng
            ).item()
            first_seed = int(pool[first_pos].item())
            seeds = [first_seed]
            x_sub = x[pool]
            dists = (1.0 - (x_sub @ x[first_seed].unsqueeze(1)).squeeze()).clamp_min(
                0.0
            )
            used_mask = torch.zeros(pool.numel(), dtype=torch.bool, device=device)
            used_mask[first_pos] = True
            for _ in range(1, num):
                probs = (dists + 1e-6) / (dists.sum() + 1e-6)
                pick_pos = torch.multinomial(
                    probs, num_samples=1, replacement=False
                ).item()
                while used_mask[pick_pos]:
                    pick_pos = (pick_pos + 1) % pool.numel()
                used_mask[pick_pos] = True
                pick = int(pool[pick_pos].item())
                seeds.append(pick)
                new_d = (1.0 - (x_sub @ x[pick].unsqueeze(1)).squeeze()).clamp_min(0.0)
                dists = torch.minimum(dists, new_d)
            return seeds

        labeled_target = min(
            labeled_idx.numel(),
            max(1, int(S * labeled_seed_ratio)),
        )
        unlabeled_target = max(0, S - labeled_target)

        if unlabeled_idx.numel() == 0:
            unlabeled_target = 0
            labeled_target = min(S, labeled_idx.numel())
        else:
            unlabeled_target = max(1, unlabeled_target)
            unlabeled_target = min(unlabeled_target, unlabeled_idx.numel())
            labeled_target = min(max(1, S - unlabeled_target), labeled_idx.numel())

        total = labeled_target + unlabeled_target
        if total < S:
            remaining = S - total
            extra_from_labeled = min(remaining, labeled_idx.numel() - labeled_target)
            labeled_target += extra_from_labeled
            remaining -= extra_from_labeled
            if remaining > 0:
                extra_from_unlabeled = min(
                    remaining, unlabeled_idx.numel() - unlabeled_target
                )
                unlabeled_target += extra_from_unlabeled
                remaining -= extra_from_unlabeled
            if remaining > 0:
                labeled_target += remaining
                labeled_target = min(labeled_target, labeled_idx.numel())

        labeled_seeds = _farthest_sampling(labeled_idx, labeled_target)
        unlabeled_seeds = _farthest_sampling(unlabeled_idx, unlabeled_target)

        if len(labeled_seeds) + len(unlabeled_seeds) == 0:
            logger.warning("方法 H: 无法采样足够种子，回退至方法 D。")
            return self.partition_caop(
                data, S=S, s_max=s_max, cache_path=cache_path, logger=logger
            )

        logger.info(
            f"方法 H: 采样到 {len(labeled_seeds)} 个已知标签种子，{len(unlabeled_seeds)} 个未知标签种子"
        )

        labeled_seed_set = set(labeled_seeds)
        remaining_anchor_candidates = [
            idx for idx in labeled_idx.tolist() if idx not in labeled_seed_set
        ]

        def _assign_anchor(unlabeled_node: int) -> Optional[int]:
            nonlocal remaining_anchor_candidates
            candidate_list = remaining_anchor_candidates
            if candidate_list:
                cand_tensor = torch.tensor(
                    candidate_list, device=device, dtype=torch.long
                )
                sims = torch.mv(x[cand_tensor], x[unlabeled_node])
                best_pos = int(torch.argmax(sims).item())
                return candidate_list.pop(best_pos)
            if labeled_seeds:
                cand_tensor = torch.tensor(
                    labeled_seeds, device=device, dtype=torch.long
                )
                sims = torch.mv(x[cand_tensor], x[unlabeled_node])
                best_pos = int(torch.argmax(sims).item())
                return labeled_seeds[best_pos]
            return None

        groups: List[List[int]] = []
        assigned_counts = torch.zeros(n, dtype=torch.long, device=device)
        feature_dim = x.size(1)
        group_feat_sum = torch.zeros(
            (len(labeled_seeds) + len(unlabeled_seeds), feature_dim), device=device
        )
        group_deg_sum = torch.zeros(len(group_feat_sum), device=device)
        group_counts = torch.zeros(len(group_feat_sum), device=device)

        def _init_group(group_idx: int, node_list: List[int]) -> None:
            unique_nodes = sorted(set(node_list))
            groups.append(unique_nodes)
            for node in unique_nodes:
                assigned_counts[node] += 1
                group_feat_sum[group_idx] += x[node]
                group_deg_sum[group_idx] += deg[node]
                group_counts[group_idx] += 1

        for i, seed in enumerate(labeled_seeds):
            _init_group(i, [seed])

        base = len(labeled_seeds)
        for offset, seed in enumerate(unlabeled_seeds):
            anchor = _assign_anchor(seed)
            group_nodes = [seed]
            if anchor is not None:
                group_nodes.append(anchor)
            else:
                logger.warning(
                    f"方法 H: 未找到可用的标签锚点，未知种子 {seed} 将独立成为社区。"
                )
            _init_group(base + offset, group_nodes)

        S_effective = len(groups)
        logger.info(f"方法 H: 初始化完成，共 {S_effective} 个微型社区")

        K = min(8, S_effective)
        batch_size = 20000 if n > 50000 else 10000
        logger.info(f"... 分配 {n} 个节点到 {S_effective} 个社区 (s_max={s_max}) ...")

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

        logger.info("... 生成子图 ...")
        subgraphs = []
        edge_device = data.edge_index.device

        for nodes in groups:
            if not nodes:
                continue
            node_indices = torch.tensor(
                sorted(set(int(i) for i in nodes)),
                dtype=torch.long,
                device=edge_device,
            )
            sub_edge_index, _ = pyg_subgraph(
                node_indices,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )
            d = Data(
                x=data.x[node_indices],
                edge_index=sub_edge_index,
                y=data.y[node_indices],
            )
            d.num_nodes = node_indices.numel()
            d.original_node_indices = node_indices
            test_ratio = 1.0 - self.train_ratio - self.val_ratio
            y_split_key = torch.full_like(d.y, fill_value=-1)
            y_split_key[d.y == 0] = 0
            y_split_key[d.y == 1] = 1
            d.y_split_key = y_split_key
            splitter = RandomNodeSplit(
                split="train_rest",
                num_val=self.val_ratio,
                num_test=test_ratio,
                key="y_split_key",
            )
            d = splitter(d)
            delattr(d, "y_split_key")
            d.test_mask = torch.zeros(d.num_nodes, dtype=torch.bool, device=d.x.device)
            subgraphs.append(d)

        logger.info(f"... 构建完成，共 {len(subgraphs)} 个子图 ...")
        self._save_cache(subgraphs, cache_path, logger)
        return subgraphs

    # --- 方法 I: 标签引导的 CAOP (已知/未知混合种子) (修改版) ---

    def partition_label_guided_caop_modified(
        self,
        data: Data,
        S: int,
        s_max: int,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 I (CAOP法):
        基于特征相似性和度相似性的受控重叠图划分算法。

        修改点：
        1. **标签感知种子选择 (Label-Aware Seeding):**
        - 强制从非法节点中选择至少一个种子，从合法节点中选择至少一个种子。
        - 如果数据集中存在非法和合法节点，则前两个种子分别从这两个集合中选择。
        - 剩余的种子选择采用原来的 K-Means++ 风格的概率选择，但候选池排除了已选的非法和合法种子。
        - 目的：确保初始社区中心具有多样化的标签，从而引导生成的子图包含不同类别的节点。
        2. **增强的相似度计算 (Enhanced Affinity):**
        - 在计算节点与社区的相似度时，引入了对稀疏类别（非法节点）的轻微偏好。
        - 如果节点是非法节点，其特征相似度得分会增加一个小的常数 `epsilon_affinity`。
        - 目的：在不违反 `s_max` 的前提下，稍微增加非法节点被分配到多个社区的可能性，从而提高它们在最终子图中的覆盖率。
        3. **代码健壮性：**
        - 增加了对 `data.y`（标签）存在的检查。
        - 增加了对是否存在非法和合法节点的检查，以处理极端不平衡或无标签数据的情况。
        - 确保张量在正确的设备上进行计算。

        参数:
            data: 输入图数据，需要包含 data.y (标签)
            S: 目标分区数
            s_max: 每个节点的最大重叠社区数
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器

        返回:
            子图列表
        """

        # 检查缓存
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(
            f"开始执行 [方法 D：CAOP法 (修改版 - 标签感知)] (S={S}, s_max={s_max})"
        )
        device = self.device
        x = data.x.to(device)
        n = data.num_nodes

        # 检查标签是否存在
        if not hasattr(data, "y") or data.y is None:
            logger.warning(
                "CAOP (修改版) 需要 data.y 标签来进行标签感知种子选择。未找到标签，将回退到原始 CAOP 行为。"
            )
            has_labels = False
        else:
            y = data.y.to(device)
            has_labels = True

        deg = self._degree_vector(data).float().to(device)
        max_deg = deg.max().clamp_min(1.0)
        x = F.normalize(x, dim=1)

        # --- 1. 标签感知种子选择 (Label-Aware Seeding) ---
        seeds = []
        rng = torch.Generator(device="cpu")  # 用于随机选择的 CPU 生成器

        # 识别非法和合法节点的索引
        illicit_indices = (
            (y == 1).nonzero(as_tuple=True)[0].cpu()
            if has_labels
            else torch.empty(0, dtype=torch.long)
        )
        licit_indices = (
            (y == 0).nonzero(as_tuple=True)[0].cpu()
            if has_labels
            else torch.empty(0, dtype=torch.long)
        )

        seeds_set = set()

        # 强制选择非法种子
        if len(illicit_indices) > 0:
            first_illicit_idx = int(
                torch.randint(
                    low=0, high=len(illicit_indices), size=(1,), generator=rng
                ).item()
            )
            first_illicit_node = illicit_indices[first_illicit_idx].item()
            seeds.append(first_illicit_node)
            seeds_set.add(first_illicit_node)
            logger.info(f"已选择一个非法节点作为种子: {first_illicit_node}")

        # 强制选择合法种子
        if len(licit_indices) > 0 and len(seeds) < S:
            # 确保不重复选择已选的非法节点（虽然概率很小）
            available_licit = [
                idx.item() for idx in licit_indices if idx.item() not in seeds_set
            ]
            if available_licit:
                first_licit_idx = int(
                    torch.randint(
                        low=0, high=len(available_licit), size=(1,), generator=rng
                    ).item()
                )
                first_licit_node = available_licit[first_licit_idx]
                seeds.append(first_licit_node)
                seeds_set.add(first_licit_node)
                logger.info(f"已选择一个合法节点作为种子: {first_licit_node}")

        # 如果没有标签或标签类型不足，回退到随机选择第一个种子
        if len(seeds) == 0:
            first = int(torch.randint(low=0, high=n, size=(1,), generator=rng).item())
            seeds.append(first)
            seeds_set.add(first)
            logger.info(f"未找到足够的标签类型，随机选择第一个种子: {first}")

        # 准备用于后续选择的候选池
        sub_ratio = 0.2 if n > 10000 else 1.0
        all_idx = torch.arange(n)
        # 排除已经选择的种子
        remaining_idx = torch.tensor(
            [i for i in all_idx.tolist() if i not in seeds_set], dtype=torch.long
        )

        if sub_ratio < 1.0 and len(remaining_idx) > 0:
            k = max(1, int(len(remaining_idx) * sub_ratio))
            # 在剩余节点中随机采样
            sub_idx_cpu = remaining_idx[
                torch.randperm(len(remaining_idx), generator=rng)[:k]
            ]
            sub_idx = sub_idx_cpu.to(device)
        else:
            sub_idx = remaining_idx.to(device)

        if len(sub_idx) == 0 and len(seeds) < S:
            logger.warning("没有足够的剩余节点来选择所有 S 个种子。")

        # 计算初始距离：候选节点到已选种子集合的最小距离
        if len(sub_idx) > 0:
            x_sub = x[sub_idx]
            # 计算到第一个种子的距离
            dists = (1.0 - (x_sub @ x[seeds[0]].unsqueeze(1)).squeeze()).clamp_min(0.0)
            # 更新到后续已选种子的最小距离
            for i in range(1, len(seeds)):
                new_d = (1.0 - (x_sub @ x[seeds[i]].unsqueeze(1)).squeeze()).clamp_min(
                    0.0
                )
                dists = torch.minimum(dists, new_d)

        # 继续选择剩余的种子 (K-Means++ 风格)
        num_seeds_to_pick = S - len(seeds)
        for _ in range(num_seeds_to_pick):
            if len(sub_idx) == 0:
                break
            # 计算选择概率
            probs = (dists + 1e-6) / (dists.sum() + 1e-6)
            # 转移到 CPU 进行多项式采样
            probs_cpu = probs.cpu()
            pick_pos = torch.multinomial(
                probs_cpu, num_samples=1, replacement=False
            ).item()
            pick = int(sub_idx[pick_pos].item())
            seeds.append(pick)

            # 更新距离
            new_d = (1.0 - (x_sub @ x[pick].unsqueeze(1)).squeeze()).clamp_min(0)
            dists = torch.minimum(dists, new_d)

        logger.info(f"总共选择了 {len(seeds)} 个种子。")

        # 2. 初始化分组
        groups: List[List[int]] = [[s] for s in seeds]
        assigned_counts = torch.zeros(n, dtype=torch.long, device=device)
        seeds_tensor = torch.tensor(seeds, device=device, dtype=torch.long)
        assigned_counts[seeds_tensor] = 1

        # 初始化社区特征和度数之和
        group_feat_sum = torch.zeros((S, x.size(1)), device=device)
        group_deg_sum = torch.zeros(S, device=device)
        group_counts = torch.zeros(S, device=device)

        # 将种子节点的信息填入对应的社区
        for i, seed_idx in enumerate(seeds):
            group_feat_sum[i] = x[seed_idx]
            group_deg_sum[i] = deg[seed_idx]
            group_counts[i] = 1.0

        # 对于没有分配到种子的空社区 (如果 S > n)，进行填充以防除零错误
        # (在当前逻辑下，len(seeds) 应该等于 S，除非 n < S)
        if len(seeds) < S:
            for i in range(len(seeds), S):
                # 用全局平均填充，或者保持为0并处理除零
                # 这里简单处理：保持为0，后续计算均值时分母加 epsilon
                pass

        # 3. 贪婪分配节点到社区
        K = min(8, S)
        batch_size = 20000 if n > 50000 else 10000
        logger.info(f"... 分配 {n} 个节点到 {S} 个社区 (s_max={s_max}) ...")

        # --- 增强的相似度计算参数 ---
        epsilon_affinity = 0.05 if has_labels else 0.0  # 对非法节点的额外亲和力奖励

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            idx_cpu = torch.arange(start, end, dtype=torch.long)
            idx = idx_cpu.to(device)

            # 过滤掉已经达到最大重叠次数的节点
            mask_remain = assigned_counts[idx] < s_max
            if not mask_remain.any():
                continue
            idx = idx[mask_remain]

            # 计算社区的平均特征和平均度数
            # group_counts 可能为0，加上 1e-6 防止除零
            g_feat_mean = F.normalize(
                group_feat_sum / (group_counts.unsqueeze(1) + 1e-6), dim=1
            )
            g_deg_mean = (group_deg_sum / (group_counts + 1e-6)).unsqueeze(0)

            # 计算特征相似度 (Cosine Similarity)
            sim_f = (x[idx] @ g_feat_mean.t()).clamp(-1, 1)

            # 计算度数相似度
            sim_d = 1.0 - (deg[idx].unsqueeze(1) - g_deg_mean).abs() / max_deg

            # 计算总亲和力
            affinity = sim_f + sim_d

            # --- 增强的相似度计算 (Enhanced Affinity) ---
            if has_labels and epsilon_affinity > 0:
                # 找出当前批次中的非法节点
                batch_y = y[idx]
                is_illicit = batch_y == 1
                # 对非法节点的亲和力矩阵增加一个小的常数
                # 利用广播机制：is_illicit.unsqueeze(1) 形状为 (batch_size, 1)
                affinity = affinity + is_illicit.unsqueeze(1).float() * epsilon_affinity

            # 选择亲和力最高的 K 个社区作为候选
            _, topk_idx = torch.topk(affinity, k=K, dim=1)

            # 贪婪分配
            for row_idx, node_idx_tensor in enumerate(idx):
                node = node_idx_tensor.item()
                if assigned_counts[node] >= s_max:
                    continue

                cands = topk_idx[row_idx]
                for gi_tensor in cands:
                    gi = gi_tensor.item()
                    if assigned_counts[node] >= s_max:
                        break

                    # 将节点加入社区
                    groups[gi].append(node)
                    assigned_counts[node] += 1

                    # 更新社区统计信息
                    group_feat_sum[gi] += x[node]
                    group_deg_sum[gi] += deg[node]
                    group_counts[gi] += 1

        # 4. 生成子图
        logger.info("... 生成子图 ...")
        subgraphs = []
        edge_device = data.edge_index.device

        for i, nodes in enumerate(groups):
            if not nodes:
                logger.debug(f"社区 {i} 为空，跳过。")
                continue

            # 转换为排序后的节点索引张量 (确保确定性)
            node_indices = torch.tensor(
                sorted(list(set(nodes))), dtype=torch.long, device=edge_device
            )

            # 使用 PyG 的 subgraph 工具提取子图
            # relabel_nodes=True 会重新编号节点，从 0 开始
            sub_edge_index, _ = pyg_subgraph(
                node_indices,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )

            # 创建子图数据对象
            d = Data(
                x=data.x[node_indices],
                edge_index=sub_edge_index,
                y=data.y[node_indices]
                if hasattr(data, "y") and data.y is not None
                else None,
            )
            d.num_nodes = node_indices.numel()
            # 保存原始节点索引，对于后续任务（如节点映射）非常重要
            d.original_node_indices = node_indices.cpu()  # 移至 CPU 以节省显存

            # 如果存在标签，创建训练/验证/测试掩码
            if d.y is not None:
                # 计算测试集比例
                test_ratio = 1.0 - self.train_ratio - self.val_ratio
                # 确保比例和为 1，处理浮点误差
                if test_ratio < 0:
                    test_ratio = 0.0

                # 使用 RandomNodeSplit 创建掩码
                # split="train_rest" 表示优先分配训练集，剩余的给验证和测试
                y_split_key = torch.full_like(d.y, fill_value=-1)
                y_split_key[d.y == 0] = 0
                y_split_key[d.y == 1] = 1
                d.y_split_key = y_split_key
                splitter = RandomNodeSplit(
                    split="train_rest",
                    num_val=self.val_ratio,
                    num_test=test_ratio,
                    key="y_split_key",
                )
                d = splitter(d)
                delattr(d, "y_split_key")

                # RandomNodeSplit 有时可能不会创建 test_mask，确保它存在
                if not hasattr(d, "test_mask"):
                    d.test_mask = torch.zeros(
                        d.num_nodes, dtype=torch.bool, device=d.x.device
                    )
                    # 如果 val_ratio + train_ratio < 1.0，则剩余的应为测试集
                    if test_ratio > 1e-6:
                        d.test_mask = ~(d.train_mask | d.val_mask)

            subgraphs.append(d)

        logger.info(f"... 构建完成，共 {len(subgraphs)} 个非空子图 ...")

        # 5. 保存缓存
        self._save_cache(subgraphs, cache_path, logger)

        return subgraphs

    # --- 方法 E: Louvain + 模合并 + 1跳邻居 + 剪枝 ---

    def partition_modular_macro(
        self,
        data: Data,
        S_hint: int,
        s_max: int,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 E (模合并剪枝):
        在方法 B 的基础上，增加一步：先将 Louvain 得到的自然社区通过模运算合并成 S_hint 个宏社区，
        再执行 1-hop 重叠与剪枝，得到更加平衡的社区划分。

        参数:
            data: 输入图数据
            S_hint: 目标宏社区数量
            s_max: 每个节点的最大重叠社区数
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器（可选）
        """
        if S_hint <= 0:
            raise ValueError("S_hint 必须为正整数")

        # 检查缓存
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"开始执行 [方法 E：模合并剪枝] (S_hint={S_hint}, s_max={s_max})")
        num_nodes = data.num_nodes

        # 1. cugraph 加速的硬划分 (Louvain)
        G_cu = self._pyg_to_cugraph_graph(data)
        logger.info("... cugraph.louvain 运行中 ...")
        partition_df = cugraph.louvain(G_cu)
        partition_df = partition_df[0]
        partition_pd = partition_df.to_pandas()

        partition_map = {row.vertex: row.partition for row in partition_pd.itertuples()}
        S_found = partition_pd["partition"].max() + 1
        logger.info(f"... Louvain 找到了 {S_found} 个自然社区 ...")

        # 1.5 通过模运算将自然社区合并为 S_hint 个宏社区
        logger.info(f"... 使用模运算将社区合并为 {S_hint} 个宏社区 ...")
        macro_partition_map = {}
        macro_sizes = [0 for _ in range(S_hint)]
        for node_id, part_id in partition_map.items():
            macro_id = part_id % S_hint
            macro_partition_map[node_id] = macro_id
            macro_sizes[macro_id] += 1

            logger.info(
                "宏社区节点数量统计: "
                + ", ".join(
                    [f"Macro {idx}: {size}" for idx, size in enumerate(macro_sizes)]
                )
            )

        # 2. 创建 1-Hop 邻居重叠（使用宏社区）
        logger.info("... 创建 1-Hop 邻居重叠 (宏社区) ...")
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
            if node in macro_partition_map:
                initial_node_to_community_map[node].add(macro_partition_map[node])

            for neighbor in adj.get(node, set()):
                if neighbor in macro_partition_map:
                    initial_node_to_community_map[node].add(
                        macro_partition_map[neighbor]
                    )

            if len(initial_node_to_community_map[node]) > max_overlap_found:
                max_overlap_found = len(initial_node_to_community_map[node])

        logger.info(f"... 发现的最大不受控重叠 s = {max_overlap_found} ...")

        # 3. 剪枝逻辑（与方法 B 相同，但使用宏社区）
        logger.info(f"... 剪枝 {num_nodes} 个节点至 s_max={s_max} ...")
        x_cpu = data.x.cpu().numpy()

        # 3a. 计算每个宏社区的质心
        macro_communities: List[List[int]] = [[] for _ in range(S_hint)]
        for node_id, macro_id in macro_partition_map.items():
            macro_communities[macro_id].append(node_id)

        centroids = np.array(
            [
                x_cpu[nodes].mean(axis=0)
                if len(nodes) > 0
                else np.zeros(x_cpu.shape[1])
                for nodes in macro_communities
            ]
        )

        # 3b. 剪枝
        final_node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]

        for node in range(num_nodes):
            communities = list(initial_node_to_community_map[node])

            if len(communities) == 0:
                continue
            elif len(communities) <= s_max:
                final_node_to_community_map[node] = set(communities)
            else:
                node_feature = x_cpu[node].reshape(1, -1)
                community_centroids = centroids[communities]
                affinities = cosine_similarity(node_feature, community_centroids)[0]
                sorted_local_indices = np.argsort(affinities)[::-1]
                for i in range(s_max):
                    local_idx = sorted_local_indices[i]
                    global_community_id = communities[local_idx]
                    final_node_to_community_map[node].add(global_community_id)

        logger.info("... 剪枝完成，正在构建 PyG 子图 ...")
        subgraphs = self._build_pyg_subgraphs(data, final_node_to_community_map, S_hint)
        # 保存缓存
        self._save_cache(subgraphs, cache_path, logger)

        return subgraphs

    # --- 方法 F: Louvain + 宏社区 + 核心优先剪枝 ---

    def partition_core_edge(
        self,
        data: Data,
        S_F: int = 15,
        use_fraud_aware_pruning: bool = True,
        fraud_weight: float = 0.5,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 F (Louvain + 宏社区 + 核心优先剪枝):
        1. Louvain 社区发现与宏社区构建（同 E 方法）
        2. 有策略的 1-Hop 扩展与"核心优先"剪枝（s_max=1）

        参数:
            data: 输入图数据
            S_F: 目标宏社区数量（例如 15）
            use_fraud_aware_pruning: 是否使用考虑欺诈标签的剪枝策略（默认False，使用标准核心优先剪枝）
            fraud_weight: 欺诈节点的加权分数（仅在 use_fraud_aware_pruning=True 时使用）
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器（可选）

        返回:
            子图列表
        """
        # 检查缓存
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(f"开始执行 [方法 F：核心优先剪枝] (S_F={S_F})")
        num_nodes = data.num_nodes

        # ========== 步骤 1: Louvain 社区发现与宏社区构建 ==========
        logger.info("步骤 1: Louvain 社区发现与宏社区构建...")

        # 1.1 Louvain 社区发现
        G_cu = self._pyg_to_cugraph_graph(data)
        logger.info("... cugraph.louvain 运行中 ...")
        partition_df = cugraph.louvain(G_cu)
        if isinstance(partition_df, tuple):
            partition_df = partition_df[0]
        partition_pd = partition_df.to_pandas()

        partition_map = {row.vertex: row.partition for row in partition_pd.itertuples()}
        S_found = partition_pd["partition"].max() + 1
        logger.info(f"... Louvain 找到了 {S_found} 个自然社区 ...")

        # 1.2 小社区过滤
        communities: Dict[int, List[int]] = {}
        for node_id, part_id in partition_map.items():
            communities.setdefault(part_id, []).append(node_id)

        filtered_communities = {
            comm_id: nodes
            for comm_id, nodes in communities.items()
            if len(nodes) >= self.min_community_size
        }
        num_filtered = len(communities) - len(filtered_communities)
        if num_filtered > 0:
            logger.info(
                f"过滤掉 {num_filtered} 个节点数量小于 {self.min_community_size} 的社区"
            )

        communities = filtered_communities

        if len(communities) == 0:
            error_msg = (
                f"过滤后没有剩余社区（所有社区都小于 {self.min_community_size} 个节点）"
            )
            logger.error(error_msg)
            return []

        # 1.3 宏社区合并（模运算）
        logger.info(f"... 通过模运算合并为 {S_F} 个宏社区 ...")
        macro_communities: Dict[int, List[int]] = {i: [] for i in range(S_F)}
        for comm_id, nodes in communities.items():
            macro_id = comm_id % S_F
            macro_communities[macro_id].extend(nodes)

        # 移除空宏社区并转换为集合
        macro_communities_list: List[Set[int]] = []
        macro_communities_dict: Dict[int, Set[int]] = {}
        for macro_id, nodes in macro_communities.items():
            if len(nodes) > 0:
                node_set = set(nodes)
                macro_communities_list.append(node_set)
                macro_communities_dict[len(macro_communities_list) - 1] = node_set

        S_F_actual = len(macro_communities_list)
        logger.info(f"生成 {S_F_actual} 个宏社区")

        # ========== 步骤 2: 有策略的 1-Hop 扩展与"核心优先"剪枝 ==========
        logger.info("步骤 2: 1-Hop 扩展与核心优先剪枝...")

        # 构建邻接表
        adj: Dict[int, Set[int]] = {}
        src, dst = data.edge_index.cpu().numpy()
        for i in range(len(src)):
            u, v = src[i], dst[i]
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # 为每个宏社区生成候选子图（宏社区节点 + 1-hop邻居）
        candidate_subgraphs: List[Set[int]] = []
        for macro_comm in macro_communities_list:
            candidate_subgraph = set(macro_comm)  # 宏社区节点本身
            # 扩展1-hop邻居
            for node in macro_comm:
                for neighbor in adj.get(node, set()):
                    candidate_subgraph.add(neighbor)
            candidate_subgraphs.append(candidate_subgraph)

        # "核心优先"剪枝（s_max=1）
        if use_fraud_aware_pruning:
            logger.info("应用核心优先+欺诈优先剪枝 (s_max=1)...")
            final_subgraphs = self._core_priority_pruning_with_fraud(
                macro_communities_list,
                candidate_subgraphs,
                data,
                fraud_weight=fraud_weight,
                logger=logger,
            )
        else:
            logger.info("应用核心优先剪枝 (s_max=1)...")
            final_subgraphs = self._core_priority_pruning(
                macro_communities_list, candidate_subgraphs, data, logger
            )

        # 过滤空子图
        final_subgraphs = [sg for sg in final_subgraphs if len(sg) > 0]
        S_F_final = len(final_subgraphs)

        logger.info(f"剪枝后剩余 {S_F_final} 个子图")

        # ========== 步骤 3: 构建最终子图 ==========
        logger.info("步骤 3: 构建最终子图...")

        # 构建节点到社区的映射
        node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]
        for subgraph_id, subgraph_nodes in enumerate(final_subgraphs):
            for node_id in subgraph_nodes:
                if node_id < num_nodes:
                    node_to_community_map[node_id].add(subgraph_id)

        # 使用 _build_pyg_subgraphs 构建最终子图
        subgraphs = self._build_pyg_subgraphs(data, node_to_community_map, S_F_final)

        # 过滤掉节点数量小于 min_community_size 的子图
        filtered_subgraphs = [
            sg for sg in subgraphs if sg.num_nodes >= self.min_community_size
        ]
        num_filtered = len(subgraphs) - len(filtered_subgraphs)

        if num_filtered > 0:
            logger.info(
                f"过滤掉 {num_filtered} 个节点数量小于 {self.min_community_size} 的子图"
            )

        if len(filtered_subgraphs) == 0:
            error_msg = (
                f"过滤后没有剩余子图（所有子图都小于 {self.min_community_size} 个节点）"
            )
            logger.error(error_msg)
            return []

        logger.info(f"方法 F 完成: 最终生成 {len(filtered_subgraphs)} 个子图")

        # 保存缓存
        self._save_cache(filtered_subgraphs, cache_path, logger)

        return filtered_subgraphs

    # --- 方法 G: 全局最优剪枝 (两阶段方法) ---

    def _generate_d_candidates(
        self,
        data: Data,
        N_D: int = 20,
        s_max_D_temp: int = 4,
        logger=None,
    ) -> List[Set[int]]:
        """
        生成D类候选子图（基于CAOP方法，使用更高的s_max）。

        参数:
            data: 输入图数据
            N_D: 生成的候选子图数量
            s_max_D_temp: 临时s_max值（用于生成候选，不限制）
            logger: 日志记录器（可选）

        返回:
            候选子图列表（每个元素是节点ID的集合）
        """
        logger.info(f"生成D类候选子图 (N_D={N_D}, s_max_D_temp={s_max_D_temp})...")

        # 使用CAOP方法生成候选子图（不限制s_max）
        caop_subgraphs = self.partition_caop(
            data, S=N_D, s_max=s_max_D_temp, cache_path=None, logger=logger
        )

        # 转换为节点集合列表
        d_candidates = []
        for sg in caop_subgraphs:
            if hasattr(sg, "original_node_indices"):
                node_set = set(sg.original_node_indices.cpu().tolist())
            else:
                node_set = set(range(sg.num_nodes))
            if len(node_set) > 0:
                d_candidates.append(node_set)

        logger.info(f"生成 {len(d_candidates)} 个D类候选子图")

        return d_candidates

    def _generate_e_candidates(
        self,
        data: Data,
        N_E: int = 20,
        logger=None,
    ) -> List[Set[int]]:
        """
        生成E类候选子图（基于E方法，但不做s_max限制）。

        参数:
            data: 输入图数据
            N_E: 生成的候选子图数量（作为宏社区数量）
            logger: 日志记录器（可选）

        返回:
            候选子图列表（每个元素是节点ID的集合）
        """
        logger.info(f"生成E类候选子图 (N_E={N_E})...")

        # 1. Louvain社区发现
        G_cu = self._pyg_to_cugraph_graph(data)
        partition_df = cugraph.louvain(G_cu)
        if isinstance(partition_df, tuple):
            partition_df = partition_df[0]
        partition_pd = partition_df.to_pandas()

        partition_map = {row.vertex: row.partition for row in partition_pd.itertuples()}

        # 2. 过滤小社区
        communities: Dict[int, List[int]] = {}
        for node_id, part_id in partition_map.items():
            communities.setdefault(part_id, []).append(node_id)

        filtered_communities = {
            comm_id: nodes
            for comm_id, nodes in communities.items()
            if len(nodes) >= self.min_community_size
        }

        if len(filtered_communities) == 0:
            logger.warning("过滤后没有剩余社区")
            return []

        # 3. 模运算合并为N_E个宏社区
        macro_communities: Dict[int, List[int]] = {i: [] for i in range(N_E)}
        for comm_id, nodes in filtered_communities.items():
            macro_id = comm_id % N_E
            macro_communities[macro_id].extend(nodes)

        # 4. 1-hop扩展（不做s_max限制）
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
            # 扩展1-hop邻居
            expanded_nodes = set(macro_nodes)
            for node in macro_nodes:
                for neighbor in adj.get(node, set()):
                    expanded_nodes.add(neighbor)
            if len(expanded_nodes) > 0:
                e_candidates.append(expanded_nodes)

        logger.info(f"生成 {len(e_candidates)} 个E类候选子图")

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
        全局最优剪枝：在全局max_overlap_per_node限制下，从候选子图池中选择最优子图。

        参数:
            candidate_pool: 候选子图池（每个元素是节点ID的集合）
            data: 输入图数据
            max_overlap_per_node: 每个节点的最大重叠数（全局限制）
            min_subgraph_size: 最小子图大小
            logger: 日志记录器（可选）

        返回:
            剪枝后的子图列表
        """
        logger.info(
            f"开始全局最优剪枝 (候选子图数={len(candidate_pool)}, max_overlap={max_overlap_per_node})..."
        )

        num_nodes = data.num_nodes

        # 获取欺诈标签
        fraud_labels = None
        if hasattr(data, "y") and data.y is not None:
            y_cpu = data.y.cpu().numpy()
            if y_cpu.dtype in [np.int64, np.int32]:
                fraud_labels = (y_cpu == 1).astype(np.float32)
            else:
                fraud_labels = (y_cpu > 0.5).astype(np.float32)

        # 1. 对候选子图进行排序（按欺诈节点比例、大小等）
        def score_subgraph(subgraph_nodes: Set[int]) -> float:
            """计算子图评分（用于排序）"""
            if len(subgraph_nodes) == 0:
                return 0.0

            # 基础分数：子图大小
            size_score = len(subgraph_nodes) / num_nodes

            # 欺诈节点比例分数
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

            # 综合评分：欺诈比例权重更高
            return fraud_score * 0.7 + size_score * 0.3

        sorted_candidates = sorted(candidate_pool, key=score_subgraph, reverse=True)

        # 2. 迭代式贪心分配
        final_subgraphs: List[Set[int]] = []
        global_overlap_counts = np.zeros(num_nodes, dtype=np.int32)

        # 记录核心节点（D方法的种子节点和E方法的宏社区节点）
        core_nodes_set = set()
        for candidate in candidate_pool:
            # 简单假设：前几个节点是核心节点（实际应该从CAOP和E方法中获取）
            if len(candidate) > 0:
                core_nodes_set.add(list(candidate)[0])

        for candidate in sorted_candidates:
            # 构建临时子图
            temp_subgraph = set(candidate)

            # 节点过滤：移除已达到max_overlap的节点
            nodes_to_remove = []
            for node_id in temp_subgraph:
                if node_id >= num_nodes:
                    nodes_to_remove.append(node_id)
                elif global_overlap_counts[node_id] >= max_overlap_per_node:
                    nodes_to_remove.append(node_id)

            for node_id in nodes_to_remove:
                temp_subgraph.remove(node_id)

            # 如果过滤后子图太小，则舍弃
            if len(temp_subgraph) < min_subgraph_size:
                continue

            # 特殊权重处理：欺诈节点和核心节点优先
            # 对于接近max_overlap的欺诈节点，如果还有预算，优先保留
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

                # 欺诈节点且还有预算，或核心节点且还有预算
                if (is_fraud and overlap_count < max_overlap_per_node) or (
                    is_core and overlap_count < max_overlap_per_node
                ):
                    priority_nodes.append(node_id)
                else:
                    normal_nodes.append(node_id)

            # 优先添加优先级节点
            for node_id in priority_nodes:
                if global_overlap_counts[node_id] < max_overlap_per_node:
                    global_overlap_counts[node_id] += 1

            # 添加正常节点（如果还有预算）
            for node_id in normal_nodes:
                if global_overlap_counts[node_id] < max_overlap_per_node:
                    global_overlap_counts[node_id] += 1

            # 更新temp_subgraph为实际可添加的节点
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
            f"全局剪枝完成: 从 {len(candidate_pool)} 个候选子图中选择 {len(final_subgraphs)} 个子图"
        )

        return final_subgraphs

    def partition_global_optimal(
        self,
        data: Data,
        N_D: int = 20,
        N_E: int = 20,
        s_max_D_temp: int = 4,
        max_overlap_per_node: int = 3,
        min_subgraph_size: int = 10,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 G (全局最优剪枝):
        Phase 1: 生成多样化的候选子图集合（D类 + E类）
        Phase 2: 全局最优剪枝与节点分配（max_overlap_per_node=3）

        参数:
            data: 输入图数据
            N_D: D类候选子图数量（15-20）
            N_E: E类候选子图数量（15-20）
            s_max_D_temp: D类候选子图生成时的临时s_max（3-4）
            max_overlap_per_node: 全局最大重叠数（默认3）
            min_subgraph_size: 最小子图大小（默认10）
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器（可选）

        返回:
            子图列表
        """
        # 检查缓存
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        logger.info(
            f"开始执行 [方法 G：全局最优剪枝] (N_D={N_D}, N_E={N_E}, max_overlap={max_overlap_per_node})"
        )
        num_nodes = data.num_nodes

        # ========== Phase 1: 生成多样化的候选子图集合 ==========
        logger.info("Phase 1: 生成多样化的候选子图集合...")

        # 1.1 生成D类候选子图
        d_candidates = self._generate_d_candidates(
            data, N_D=N_D, s_max_D_temp=s_max_D_temp, logger=logger
        )

        # 1.2 生成E类候选子图
        e_candidates = self._generate_e_candidates(data, N_E=N_E, logger=logger)

        # 1.3 合并形成候选子图池
        candidate_pool = d_candidates + e_candidates

        logger.info(
            f"候选子图池生成完成: D类={len(d_candidates)}, E类={len(e_candidates)}, 总计={len(candidate_pool)}"
        )

        # ========== Phase 2: 全局最优剪枝与节点分配 ==========
        logger.info("Phase 2: 全局最优剪枝与节点分配...")

        final_subgraphs = self._global_optimal_pruning(
            candidate_pool,
            data,
            max_overlap_per_node=max_overlap_per_node,
            min_subgraph_size=min_subgraph_size,
            logger=logger,
        )

        if len(final_subgraphs) == 0:
            error_msg = "全局剪枝后没有剩余子图"
            logger.error(error_msg)
            return []

        # ========== 步骤 3: 构建最终子图 ==========
        logger.info("步骤 3: 构建最终子图...")

        # 构建节点到社区的映射
        node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]
        for subgraph_id, subgraph_nodes in enumerate(final_subgraphs):
            for node_id in subgraph_nodes:
                if node_id < num_nodes:
                    node_to_community_map[node_id].add(subgraph_id)

        # 使用 _build_pyg_subgraphs 构建最终子图
        subgraphs = self._build_pyg_subgraphs(
            data, node_to_community_map, len(final_subgraphs)
        )

        # 过滤掉节点数量小于 min_community_size 的子图
        filtered_subgraphs = [
            sg for sg in subgraphs if sg.num_nodes >= self.min_community_size
        ]
        num_filtered = len(subgraphs) - len(filtered_subgraphs)

        if num_filtered > 0:
            logger.info(
                f"过滤掉 {num_filtered} 个节点数量小于 {self.min_community_size} 的子图"
            )

        if len(filtered_subgraphs) == 0:
            error_msg = (
                f"过滤后没有剩余子图（所有子图都小于 {self.min_community_size} 个节点）"
            )
            logger.error(error_msg)
            return []

        logger.info(f"方法 G 完成: 最终生成 {len(filtered_subgraphs)} 个子图")

        # 保存缓存
        self._save_cache(filtered_subgraphs, cache_path, logger)

        return filtered_subgraphs

    def partition_label_balanced_global(
        self,
        data: Data,
        N_D: int = 20,
        N_E: int = 20,
        s_max_D_temp: int = 4,
        max_overlap_per_node: int = 3,
        min_subgraph_size: int = 10,
        cache_path: Optional[str] = None,
        logger=None,
    ) -> List[Data]:
        """
        方法 J (标签均衡全局剪枝):
        在方法 G 的基础上，保留 Phase 1~2 的流程，改为在构建最终 PyG 子图后才进行标签均衡过滤。
        仅保留同时包含 y=0 与 y=1 的子图，且其 train/val 集也需各自包含两类标签。
        """
        cached = self._load_cache(cache_path, logger)
        if cached is not None:
            return cached

        if not hasattr(data, "y") or data.y is None:
            logger.warning("方法 J: 数据缺少标签信息，回退至方法 G。")
            return self.partition_global_optimal(
                data,
                N_D=N_D,
                N_E=N_E,
                s_max_D_temp=s_max_D_temp,
                max_overlap_per_node=max_overlap_per_node,
                min_subgraph_size=min_subgraph_size,
                cache_path=cache_path,
                logger=logger,
            )

        labels = data.y.view(-1).cpu()
        if not ((labels == 0).any() and (labels == 1).any()):
            logger.warning(
                "方法 J: 标签中缺少 0 或 1，无法执行标签均衡过滤，回退至方法 G。"
            )
            return self.partition_global_optimal(
                data,
                N_D=N_D,
                N_E=N_E,
                s_max_D_temp=s_max_D_temp,
                max_overlap_per_node=max_overlap_per_node,
                min_subgraph_size=min_subgraph_size,
                cache_path=cache_path,
                logger=logger,
            )

        logger.info(
            f"开始执行 [方法 J：标签均衡全局剪枝] (N_D={N_D}, N_E={N_E}, max_overlap={max_overlap_per_node})"
        )
        num_nodes = data.num_nodes

        # Phase 1 与方法 G 相同
        logger.info("Phase 1: 生成多样化的候选子图集合 (标签过滤前)...")
        d_candidates = self._generate_d_candidates(
            data, N_D=N_D, s_max_D_temp=s_max_D_temp, logger=logger
        )
        e_candidates = self._generate_e_candidates(data, N_E=N_E, logger=logger)
        candidate_pool = d_candidates + e_candidates

        logger.info(
            f"候选子图池生成完成: D类={len(d_candidates)}, E类={len(e_candidates)}, 总计={len(candidate_pool)}"
        )

        # Phase 2: 全局最优剪枝
        logger.info("Phase 2: 全局最优剪枝与节点分配 ...")
        final_subgraphs = self._global_optimal_pruning(
            candidate_pool,
            data,
            max_overlap_per_node=max_overlap_per_node,
            min_subgraph_size=min_subgraph_size,
            logger=logger,
        )

        if len(final_subgraphs) == 0:
            logger.error("方法 J: 全局剪枝后没有剩余子图")
            return []

        # Phase 3: 构建最终子图
        logger.info("步骤 3: 构建最终子图 ...")
        node_to_community_map: List[Set[int]] = [set() for _ in range(num_nodes)]
        for subgraph_id, subgraph_nodes in enumerate(final_subgraphs):
            for node_id in subgraph_nodes:
                if node_id < num_nodes:
                    node_to_community_map[node_id].add(subgraph_id)

        subgraphs = self._build_pyg_subgraphs(
            data, node_to_community_map, len(final_subgraphs)
        )

        balanced_subgraphs = self._filter_label_balanced_subgraphs(subgraphs, logger)
        if len(balanced_subgraphs) == 0:
            logger.error("方法 J: 标签均衡过滤后没有剩余子图")
            return []

        logger.info(f"方法 J 完成: 最终生成 {len(balanced_subgraphs)} 个子图")
        self._save_cache(balanced_subgraphs, cache_path, logger)
        return balanced_subgraphs

    def _filter_label_balanced_subgraphs(
        self, subgraphs: List[Data], logger=None
    ) -> List[Data]:
        """
        过滤子图，要求：
            1. 节点数不少于 self.min_community_size；
            2. 至少包含一条边；
            3. 整体 y 同时包含 0/1；
            4. train_mask 与 val_mask 均包含 0/1。
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
                    f"过滤掉 {removed_small} 个节点数量小于 {self.min_community_size} 的子图"
                )
            if removed_edge > 0:
                logger.info(f"过滤掉 {removed_edge} 个没有边的子图")
            if removed_label > 0:
                logger.info(f"过滤掉 {removed_label} 个整体标签不含 0/1 的子图")
            if removed_train > 0:
                logger.info(f"过滤掉 {removed_train} 个训练集标签不含 0/1 的子图")
            if removed_val > 0:
                logger.info(f"过滤掉 {removed_val} 个验证集标签不含 0/1 的子图")

        return balanced

    def compute_tsv(
        self,
        private_data: Data,
        subgraphs: List[Data],
        cache_path: Optional[str] = None,
        logger=None,
    ) -> torch.Tensor:
        """
        计算 TSV (Task-Specific Vector) 特征向量。

        参数:
            private_data: 私有数据图
            subgraphs: 子图列表
            cache_path: 缓存文件路径（如果为 None 则不使用缓存）
            logger: 日志记录器（可选）

        返回:
            TSV 张量，形状为 [num_subgraphs, 6]
        """
        num_subgraphs = len(subgraphs)
        device = self.device

        # 检查缓存
        if cache_path is not None and os.path.exists(cache_path):
            try:
                cached_tsv = torch.load(cache_path)
                # 检查缓存的 TSV 是否为 Tensor 且尺寸匹配
                if isinstance(cached_tsv, torch.Tensor):
                    # TSV 形状应该是 [num_communities, 6]
                    if cached_tsv.dim() == 2 and cached_tsv.shape[0] == num_subgraphs:
                        logger.info(
                            f"从缓存加载 TSV: shape={tuple(cached_tsv.shape)}, "
                            f"与子图数量 {num_subgraphs} 匹配"
                        )
                        return cached_tsv.to(device)
                    else:
                        logger.warning(
                            f"缓存的 TSV 尺寸不匹配: shape={tuple(cached_tsv.shape)}, "
                            f"期望第一维为 {num_subgraphs}，将重新计算"
                        )
                else:
                    logger.warning(
                        f"缓存的 TSV 类型不正确: {type(cached_tsv)}，将重新计算"
                    )
            except Exception as e:
                logger.warning(f"加载 TSV 缓存失败: {e}，将重新计算")

        # 在整个隐私图上计算 Pagerank、Clustering 和 Degree
        logger.info("在整个隐私图上计算 Pagerank、Clustering 和 Degree...")

        # 1. 计算 Degree
        deg = self._degree_vector(private_data).float().to(device)

        # 2. 计算 Pagerank 和 Clustering（使用 cugraph）
        pagerank = torch.zeros(
            private_data.num_nodes, dtype=torch.float32, device=device
        )
        clustering = torch.zeros(
            private_data.num_nodes, dtype=torch.float32, device=device
        )

        try:
            # 转换为 cugraph 格式
            edges = private_data.edge_index.cpu().numpy()
            df = cudf.DataFrame()
            df["src"] = edges[0].astype(np.int32)
            df["dst"] = edges[1].astype(np.int32)

            # 创建 cugraph 图
            G_cu = cugraph.Graph()
            G_cu.from_cudf_edgelist(df, source="src", destination="dst")

            # 计算 Pagerank
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
                    # 确保索引不越界
                    valid_mask = pagerank_indices < private_data.num_nodes
                    if valid_mask.any():
                        pagerank[pagerank_indices[valid_mask]] = pagerank_values[
                            valid_mask
                        ]
            except Exception as e:
                logger.warning(f"计算 Pagerank 时出错: {e}")

            # 计算 Clustering
            try:
                triangle_counts = cugraph.triangle_count(G_cu)
                triangle_counts_df = triangle_counts.to_pandas()
                degrees = G_cu.degree()
                degrees_df = degrees.to_pandas()

                if len(triangle_counts_df) > 0 and len(degrees_df) > 0:
                    # 合并三角形计数和度数信息
                    merged_df = triangle_counts_df.merge(
                        degrees_df, left_on="vertex", right_on="vertex", how="outer"
                    )
                    merged_df = merged_df.fillna(0)

                    # 计算每个节点的聚类系数: 2 * triangle_count / (degree * (degree-1))
                    merged_df["clustering_coeff"] = merged_df.apply(
                        lambda row: 2
                        * row["counts"]
                        / (row["degree"] * (row["degree"] - 1))
                        if row["degree"] > 1
                        else 0.0,
                        axis=1,
                    )

                    # 填充聚类系数
                    for _, row in merged_df.iterrows():
                        node = int(row["vertex"])
                        if node < private_data.num_nodes:
                            clustering[node] = float(row["clustering_coeff"])
            except Exception as e:
                logger.warning(f"计算 Clustering 时出错: {e}")
        except Exception as e:
            logger.warning(f"使用 cugraph 计算图指标时出错: {e}，将使用默认值 0")

        # 3. 计算每个子图的 TSV
        tsv_list = []
        for sg in subgraphs:
            # 获取原始节点索引
            if hasattr(sg, "original_node_indices"):
                original_indices = sg.original_node_indices.to(device)
            else:
                # 如果没有保存原始索引，使用子图的节点索引（向后兼容）
                original_indices = torch.arange(sg.num_nodes, device=device)

            if original_indices.numel() == 0:
                tsv_list.append(torch.zeros(6, device=device))
                continue

            # 从整个图中提取节点特征和标签
            x = private_data.x[original_indices].to(device)
            y = private_data.y[original_indices].float().to(device)

            # 计算子图指标
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

        # 保存缓存
        if cache_path is not None:
            os.makedirs(
                os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
                exist_ok=True,
            )
            try:
                torch.save(tsv, cache_path)
                logger.info(f"TSV 已保存到缓存: {cache_path}")
            except Exception as e:
                logger.warning(f"保存 TSV 缓存失败: {e}")

        return tsv.to(device)
