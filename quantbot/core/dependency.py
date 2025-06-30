"""
依赖关系管理模块

实现拓扑排序算法处理因子依赖关系
"""

from typing import Dict, List, Set
from collections import defaultdict, deque


class DependencyManager:
    """
    依赖关系管理器
    
    使用有向无环图(DAG)管理因子之间的依赖关系
    提供拓扑排序功能确定计算顺序
    """
    
    def __init__(self):
        # 邻接表表示图
        self.graph: Dict[str, List[str]] = defaultdict(list)
        # 入度统计
        self.in_degree: Dict[str, int] = defaultdict(int)
        # 所有节点集合
        self.nodes: Set[str] = set()
    
    def add_node(self, node: str) -> None:
        """
        添加节点
        
        Args:
            node: 节点名称
        """
        if node not in self.nodes:
            self.nodes.add(node)
            if node not in self.in_degree:
                self.in_degree[node] = 0
    
    def add_dependency(self, dependent: str, dependency: str) -> None:
        """
        添加依赖关系
        
        Args:
            dependent: 依赖者（需要依赖其他节点的节点）
            dependency: 被依赖者（被其他节点依赖的节点）
        """
        # 确保两个节点都存在
        self.add_node(dependent)
        self.add_node(dependency)
        
        # 添加边：dependency -> dependent
        if dependent not in self.graph[dependency]:
            self.graph[dependency].append(dependent)
            self.in_degree[dependent] += 1
    
    def remove_dependency(self, dependent: str, dependency: str) -> None:
        """
        移除依赖关系
        
        Args:
            dependent: 依赖者
            dependency: 被依赖者
        """
        if dependency in self.graph and dependent in self.graph[dependency]:
            self.graph[dependency].remove(dependent)
            self.in_degree[dependent] -= 1
    
    def has_cycle(self) -> bool:
        """
        检查是否存在循环依赖
        
        Returns:
            如果存在循环依赖返回True，否则返回False
        """
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True
    
    def topological_sort(self) -> List[str]:
        """
        拓扑排序
        
        使用Kahn算法实现拓扑排序
        
        Returns:
            拓扑排序后的节点列表
            
        Raises:
            ValueError: 如果存在循环依赖
        """
        # 复制入度信息，避免修改原始数据
        in_degree_copy = self.in_degree.copy()
        
        # 初始化队列，包含所有入度为0的节点
        queue = deque()
        for node in self.nodes:
            if in_degree_copy[node] == 0:
                queue.append(node)
        
        result = []
        
        while queue:
            # 取出一个入度为0的节点
            current = queue.popleft()
            result.append(current)
            
            # 遍历当前节点的所有邻接节点
            for neighbor in self.graph[current]:
                in_degree_copy[neighbor] -= 1
                # 如果邻接节点的入度变为0，加入队列
                if in_degree_copy[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否所有节点都被处理
        if len(result) != len(self.nodes):
            remaining_nodes = [node for node in self.nodes if node not in result]
            raise ValueError(f"Circular dependency detected among nodes: {remaining_nodes}")
        
        return result
    
    def get_layers(self) -> List[List[str]]:
        """
        获取分层结构
        
        将节点按依赖层级分组，同一层的节点可以并行计算
        
        Returns:
            分层后的节点列表，每一层是一个列表
        """
        # 复制入度信息
        in_degree_copy = self.in_degree.copy()
        layers = []
        
        while True:
            # 找到当前层所有入度为0的节点
            current_layer = []
            for node in self.nodes:
                if node not in [n for layer in layers for n in layer] and in_degree_copy[node] == 0:
                    current_layer.append(node)
            
            if not current_layer:
                break
                
            layers.append(current_layer)
            
            # 更新入度信息
            for node in current_layer:
                for neighbor in self.graph[node]:
                    in_degree_copy[neighbor] -= 1
        
        # 检查是否所有节点都被处理
        processed_nodes = [node for layer in layers for node in layer]
        if len(processed_nodes) != len(self.nodes):
            remaining_nodes = [node for node in self.nodes if node not in processed_nodes]
            raise ValueError(f"Circular dependency detected among nodes: {remaining_nodes}")
        
        return layers
    
    def get_dependencies(self, node: str) -> List[str]:
        """
        获取指定节点的所有直接依赖
        
        Args:
            node: 节点名称
            
        Returns:
            依赖列表
        """
        dependencies = []
        for dependency, dependents in self.graph.items():
            if node in dependents:
                dependencies.append(dependency)
        return dependencies
    
    def get_dependents(self, node: str) -> List[str]:
        """
        获取依赖于指定节点的所有直接依赖者
        
        Args:
            node: 节点名称
            
        Returns:
            依赖者列表
        """
        return self.graph[node].copy()
    
    def get_all_dependencies(self, node: str) -> Set[str]:
        """
        获取指定节点的所有依赖（包括间接依赖）
        
        Args:
            node: 节点名称
            
        Returns:
            所有依赖的集合
        """
        visited = set()
        stack = [node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            # 添加直接依赖
            direct_deps = self.get_dependencies(current)
            for dep in direct_deps:
                if dep not in visited:
                    stack.append(dep)
        
        # 移除自身
        visited.discard(node)
        return visited
    
    def get_all_dependents(self, node: str) -> Set[str]:
        """
        获取依赖于指定节点的所有节点（包括间接依赖者）
        
        Args:
            node: 节点名称
            
        Returns:
            所有依赖者的集合
        """
        visited = set()
        stack = [node]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            # 添加直接依赖者
            direct_dependents = self.get_dependents(current)
            for dependent in direct_dependents:
                if dependent not in visited:
                    stack.append(dependent)
        
        # 移除自身
        visited.discard(node)
        return visited
    
    def clear(self) -> None:
        """清空所有数据"""
        self.graph.clear()
        self.in_degree.clear()
        self.nodes.clear()
    
    def get_graph_info(self) -> Dict[str, any]:
        """
        获取图的信息
        
        Returns:
            包含图信息的字典
        """
        return {
            'nodes': list(self.nodes),
            'edges': [(dep, node) for dep, nodes in self.graph.items() for node in nodes],
            'node_count': len(self.nodes),
            'edge_count': sum(len(nodes) for nodes in self.graph.values()),
            'in_degrees': dict(self.in_degree)
        }
    
    def __repr__(self) -> str:
        return f"DependencyManager(nodes={len(self.nodes)}, edges={sum(len(deps) for deps in self.graph.values())})"