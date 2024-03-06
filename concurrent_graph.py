from typing import Hashable, Any, Optional, List, Tuple
from multiprocessing import Lock, Array, Manager, Queue, Process, Event, Value
import random
import time
import cProfile


class ConcurrentGraph:
    """
    A class representing a concurrent graph that supports thread-safe operations and parallel BFS traversal.

    The ConcurrentGraph class provides a way to represent a graph data structure in a thread-safe manner,
    allowing for concurrent modifications and queries. One of the most interesting functionalities of this
    class is the `multiprocessing_bfs` method, which performs a breadth-first search (BFS) traversal of the
    graph using multiple processes for faster execution on large graphs.

    Attributes:
        graph (dict): A dictionary representing the graph, where keys are vertices and values are dictionaries
                      of adjacent vertices with edge weights.
        lock (Lock): A threading lock to ensure thread-safe operations on the graph.

    Methods:
        add_vertex(vertex): Adds a vertex to the graph.
        add_edge(vertex1, vertex2, weight): Adds an edge between two vertices with an optional weight.
        remove_vertex(vertex): Removes a vertex and its associated edges from the graph.
        remove_edge(vertex1, vertex2): Removes an edge between two vertices.
        generate_connected_graph(num_vertices): Generates a connected graph with a specified number of vertices.
        get_vertices(): Returns a list of all vertices in the graph.
        get_edges(): Returns a list of all edges in the graph.
        get_adjacent_vertices(vertex): Returns a list of vertices adjacent to a given vertex.
        get_degree(vertex): Returns the degree of a given vertex.
        is_connected(vertex1, vertex2): Checks if there is a connection between two vertices.
        multiprocessing_bfs(start_vertex, num_processes): Performs a parallel BFS traversal of the graph.
        bfs(start_vertex): Performs a regular BFS traversal of the graph.
        __str__(): Returns a string representation of the graph.
    """

    def __init__(self):
        """
        Initialize a new instance of ConcurrentGraph.

        Creates an empty graph represented as a dictionary and initializes a lock for thread-safe operations on the graph.
        """
        self.graph = {}
        self.lock = Lock()

    def add_vertex(self, vertex: Hashable) -> bool:
        """Add a vertex to the graph.

        Args:
            vertex (Hashable): The vertex to be added to the graph.

        Returns:
            bool: True if the vertex was successfully added, False if the vertex already exists in the graph.
        """
        with self.lock:
            if vertex in self.graph:
                return False

            self.graph[vertex] = {}
            return True

    def add_edge(self, vertex1: Hashable, vertex2: Hashable, weight: float = 0) -> bool:
        """Add a new edge between vertex1 and vertex2 with the given weight.

        Args:
            vertex1 (Hashable): The first vertex to connect.
            vertex2 (Hashable): The second vertex to connect.
            weight (float): The weight of the edge. Default is 0.

        Returns:
            bool: True if the edge was successfully added, False if either vertex does not exist in the graph or the edge already exists.
        """
        with self.lock:
            if vertex1 not in self.graph or vertex2 not in self.graph:
                return False

            if vertex2 in self.graph[vertex1]:
                return False

            self.graph[vertex1][vertex2] = weight
            return True

    def remove_vertex(self, vertex: Hashable) -> Optional[dict]:
        """Remove a vertex and its associated edges from the graph.

        Args:
            vertex (Hashable): The vertex to be removed.

        Returns:
            Optional[dict]: A dictionary mapping adjacent vertices to edge weights,
                            if the vertex was present in the graph; otherwise, None.
        """
        with self.lock:
            # Remove edges from other vertices pointing to this vertex
            for v in self.graph:
                if vertex in self.graph[v]:
                    del self.graph[v][vertex]

            # Remove the vertex and its associated edges
            return self.graph.pop(vertex, None)

    def remove_edge(self, vertex1: Hashable, vertex2: Hashable) -> bool:
        """Remove an edge between vertex1 and vertex2 from the graph.

        Args:
            vertex1 (Hashable): The first vertex of the edge to be removed.
            vertex2 (Hashable): The second vertex of the edge to be removed.

        Returns:
            bool: True if the edge was successfully removed, False otherwise.
        """
        with self.lock:
            if vertex1 in self.graph and vertex2 in self.graph[vertex1]:
                del self.graph[vertex1][vertex2]
                return True
            return False

    def generate_connected_graph(self, num_vertices: int) -> None:
        """Generate a connected graph with the specified number of vertices.

        Args:
            num_vertices (int): The number of vertices to include in the graph.
        """
        # Add vertices to the graph
        for vertex in range(num_vertices):
            self.add_vertex(vertex)

        # Ensure the graph is connected by creating a path
        for vertex in range(1, num_vertices):
            self.add_edge(vertex - 1, vertex)

        # Add more edges to make the graph more complex
        for _ in range(num_vertices * 2):
            vertex1 = random.randint(0, num_vertices - 1)
            vertex2 = random.randint(0, num_vertices - 1)
            if vertex1 != vertex2:
                self.add_edge(vertex1, vertex2)

    def get_vertices(self) -> List[Hashable]:
        """Return a list of all vertices in the graph.

        Returns:
            List[Hashable]: A list of vertices in the graph.
        """
        with self.lock:
            return list(self.graph.keys())

    def get_edges(self) -> List[Tuple[Hashable, Hashable]]:
        """Return a list of all edges in the graph.

        Returns:
            List[Tuple[Hashable, Hashable]]: A list of tuples representing the edges in the graph.
        """
        with self.lock:
            edges = []
            for vertex, neighbors in self.graph.items():
                for neighbor in neighbors:
                    edges.append((vertex, neighbor))
            return edges

    def get_adjacent_vertices(self, vertex: Hashable) -> List[Hashable]:
        """Return a list of vertices adjacent to the given vertex.

        Args:
            vertex (Hashable): The vertex for which to find adjacent vertices.

        Returns:
            List[Hashable]: A list of vertices adjacent to the given vertex.

        Raises:
            KeyError: If the given vertex is not in the graph.
        """
        with self.lock:
            if vertex not in self.graph:
                raise KeyError(f"Vertex {vertex} is not in the graph.")
            return list(self.graph[vertex].keys())

    def get_degree(self, vertex: Hashable) -> int:
        """Return the degree of the given vertex

        Args:
            vertex (Hashable): The vertex for which to find the degree

        Raises:
            KeyError: If vetrex is not in the graph

        Returns:
            int: The degree of the vertex
        """
        with self.lock:
            if vertex not in self.graph:
                raise KeyError(f"Vertex {vertex} is not in the graph.")
            return len(self.graph[vertex])

    def is_connected(self, vertex1: Hashable, vertex2: Hashable) -> bool:
        """Check if there is a connection between vertex1 and vertex2.

        For a directed graph, this checks if there is an edge from vertex1 to vertex2.
        For an undirected graph, this checks if there is an edge between vertex1 and vertex2.

        Args:
            vertex1 (Hashable): The first vertex.
            vertex2 (Hashable): The second vertex.

        Returns:
            bool: True if there is a connection between vertex1 and vertex2, False otherwise.

        Raises:
            KeyError: If either vertex1 or vertex2 is not in the graph.
        """
        with self.lock:
            if vertex1 not in self.graph:
                raise KeyError(f"Vertex {vertex1} is not in the graph.")
            if vertex2 not in self.graph:
                raise KeyError(f"Vertex {vertex2} is not in the graph.")

            return vertex2 in self.graph[vertex1] or vertex1 in self.graph[vertex2]

    def multiprocessing_bfs(self, start_vertex, num_processes=4):
        """
        Perform a breadth-first search (BFS) traversal of the graph using multiple processes.

        This method uses multiprocessing to parallelize the BFS traversal, which can lead to faster
        execution times for large graphs.

        Args:
            start_vertex (Hashable): The vertex from which to start the BFS traversal.
            num_processes (int): The number of processes to use for the BFS traversal. Default is 4.

        Returns:
            List[Hashable]: A list of vertices in the order they were visited during the BFS traversal.
        """
        # Create shared objects for inter-process communication
        manager = Manager()
        visited = manager.dict()  # Shared dictionary to keep track of visited vertices
        queue = Queue()  # Queue for BFS traversal

        # Add the starting vertex to the queue
        queue.put(start_vertex)

        processes = []  # List to keep track of worker processes
        wait_events = []  # List of events to monitor worker idle states
        program_is_running = Value("b", True)  # Shared boolean to control worker loops

        # Create and start worker processes
        for _ in range(num_processes):
            wait_event = Event()
            wait_events.append(wait_event)
            p = Process(
                target=self._bfs_worker,
                args=(queue, visited, wait_event, program_is_running),
            )
            p.start()
            processes.append(p)

        # Wait for all workers to be idle before stopping
        while any(not event.is_set() for event in wait_events):
            time.sleep(1)  # Reduce sleep time for more responsive checking

        # Signal workers to stop
        program_is_running.value = False
        for _ in range(num_processes):
            queue.put(None)  # Send termination signal to each worker

        # Wait for all worker processes to terminate
        for p in processes:
            p.join()

        return list(visited)

    def _bfs_worker(self, queue, visited, wait_event, keep_working, batch_size=10000):
        """
        Worker function for parallel BFS traversal.

        This function is intended to be run as a separate process. It continuously processes vertices from a shared
        queue, performing a local BFS traversal and updating a shared dictionary of visited vertices.

        Args:
            queue (Queue): A multiprocessing queue containing vertices to be processed.
            visited (dict): A shared dictionary for keeping track of visited vertices.
            wait_event (Event): An event used to signal when the worker is waiting for new vertices.
            keep_working (Value): A shared boolean value used to control the worker loop. The worker stops when this value is False.
            batch_size (int): The number of vertices to process before transferring data back to the shared queue. Default is 10000.

        The function does not return a value but updates the shared 'visited' dictionary and 'queue' with processed vertices.
        """
        local_queue = []  # Local queue to reduce frequent access to the shared queue
        vertices_processed = 0  # Counter for processed vertices

        # Worker loop, continues until signaled to stop
        while keep_working.value:
            # Periodically move items from the local queue to the shared queue
            if (
                vertices_processed % batch_size == 0
                and vertices_processed != 0
                and local_queue
            ):
                queue.put(local_queue.pop(0))

            # If the local queue is empty, fetch a new item from the shared queue
            if not local_queue:
                wait_event.set()  # Signal that the worker is about to wait
                local_queue.append(queue.get())
                wait_event.clear()  # Signal that the worker has resumed

            # Process the next vertex in the local queue
            else:
                vertex = local_queue.pop()
                if vertex is None:  # Check for termination signal
                    break
                if vertex not in visited:  # Check if the vertex has not been visited
                    visited[vertex] = True  # Mark the vertex as visited
                    vertices_processed += 1
                    # Add adjacent vertices to the local queue
                    for adjacent_vertex in self.graph[vertex]:
                        local_queue.append(adjacent_vertex)

    def bfs(self, start_vertex: Hashable) -> List[Hashable]:
        """Perform a breadth-first search (BFS) traversal of the graph starting from the given vertex.

        Args:
            start_vertex (Hashable): The vertex from which to start the BFS traversal.

        Returns:
            List[Hashable]: A list of vertices in the order they were visited during the BFS traversal.
        """
        visited = set()  # Set to keep track of visited vertices
        queue = [
            start_vertex
        ]  # Queue for BFS traversal, initialized with the start vertex
        bfs_order = []  # List to store the order of visited vertices

        while queue:
            current_vertex = queue.pop(0)
            if current_vertex not in visited:
                visited.add(current_vertex)
                bfs_order.append(current_vertex)

                # Enqueue all adjacent vertices that haven't been visited
                for adjacent_vertex in self.graph.get(current_vertex, []):
                    if adjacent_vertex not in visited:
                        queue.append(adjacent_vertex)

        return bfs_order

    def __str__(self):
        out = ""
        for key in self.graph:
            out += str(key) + ": " + str(self.graph[key]) + "\n"
        return out


def profile_bfs_methods():
    # Set up the graph
    graph = ConcurrentGraph()
    graph.generate_connected_graph(100000)

    # Profile the regular BFS method
    print("Profiling regular BFS...")
    cProfile.runctx("graph.bfs(0)", globals(), locals())

    # Profile the multiprocessing BFS method
    print("\nProfiling multiprocessing BFS...")
    cProfile.runctx("graph.multiprocessing_bfs(0)", globals(), locals())


if __name__ == "__main__":
    profile_bfs_methods()
