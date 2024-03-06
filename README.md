# Concurrent Graph

A Python implementation of a concurrent <u>directed</u> graph data structure with support for thread-safe operations and parallel breadth-first search (BFS) traversal.

## Features

- **Thread-Safe Operations**: Supports concurrent modifications and queries on the graph.
- **Parallel BFS Traversal**: Leverages multiprocessing to perform a BFS traversal on large graphs.
- **Flexible Graph Representation**: Allows for easy addition, removal, and modification of vertices and edges.

## Installation

No external dependencies are required. Simply clone this repository or download the `concurrent_graph.py` file to your project directory.

## Usage

### Creating a Graph

```python
from concurrent_graph import ConcurrentGraph

# Initialize a new graph
graph = ConcurrentGraph()

# Add vertices
graph.add_vertex("A")
graph.add_vertex("B")

# Add an edge with a weight
graph.add_edge("A", "B", weight=1.0)
```

### Modifying the Graph

```python
# Remove an edge
graph.remove_edge("A", "B")

# Remove a vertex
graph.remove_vertex("A")
```

### Querying the Graph

```python
# Get all vertices
vertices = graph.get_vertices()

# Get all edges
edges = graph.get_edges()

# Get adjacent vertices
neighbors = graph.get_adjacent_vertices("B")

# Check if the graph is connected
is_connected = graph.is_connected("A", "B")
```

### Traversing the Graph
```python
# Perform a regular BFS traversal
bfs_order = graph.bfs(start_vertex="B")

# Perform a parallel BFS traversal using 4 processes
parallel_bfs_order = graph.multiprocessing_bfs(start_vertex="B", num_processes=4)
```


### Performance Profiling

The profile_bfs_methods function in concurrent_graph.py can be used to profile the performance of the regular and multiprocessing BFS methods:

```python
from concurrent_graph import 

profile_bfs_methods

profile_bfs_methods()
```