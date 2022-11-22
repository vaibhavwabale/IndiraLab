#BFS

graph = {
    'A': ['B','C'],
    'B': ['D','E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

visited = []  # List to keep track of visited nodes.
queue = []    # Initialize a queue

def bfs(visited, graph, node):
    visited.append(node)
    queue.append(node)
    
    while queue:
        s = queue.pop(0)
        print(s, end = " ")
        
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

# Driver Code
bfs(visited, graph, 'A')

## Output
A B C D E F

=================================================================================================

# DFS

graph = { 
    'A': set(['B','C']),
    'B': set(['D','E']),
    'C': set(['F']),
    'D': set([]),
    'E': set(['F']),
    'F': set([]),
}

def dfs(graph, start, visited = None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)
        
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited

# Driver Code
dfs(graph, 'B')

# Output
B
D
E
F
{'B', 'D', 'E', 'F'}

=================================================================================================
