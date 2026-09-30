import heapq

def predict_flood_time(current_level, rate_of_rise, danger_threshold):
    if rate_of_rise <= 0: return float('inf')
    return (danger_threshold - current_level) / rate_of_rise

def dijkstra_flood_routing(graph, start, end, flooded_nodes):
    queue = [(0, start, [])]
    seen = set()
    
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node in seen: continue
        seen.add(node)
        path = path + [node]
        
        if node == end: return (cost, path)
            
        for next_node, weight in graph[node].items():
            effective_weight = float('inf') if next_node in flooded_nodes else weight
            if next_node not in seen and effective_weight != float('inf'):
                heapq.heappush(queue, (cost + effective_weight, next_node, path))
                
    return (float('inf'), [])

road_network = {
    'Home': {'Underpass_1': 5, 'Bypass_Road': 15},
    'Underpass_1': {'College': 5},
    'Bypass_Road': {'College': 10},
    'College': {}
}

print("Normal conditions path:")
cost, path = dijkstra_flood_routing(road_network, 'Home', 'College', flooded_nodes=[])
print(f"Path: {path}, Cost: {cost} mins\n")

print("When 'Underpass_1' is Flooded:")
cost, path = dijkstra_flood_routing(road_network, 'Home', 'College', flooded_nodes=['Underpass_1'])
print(f"Path: {path}, Cost: {cost} mins")
