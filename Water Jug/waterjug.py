def water_jug_dfs(capacity_a, capacity_b, target, max_depth=1000):
    initial_state = (0, 0)
    stack = [(initial_state, [], 0)]
    visited = set([initial_state])
    
    while stack:
        state, path, depth = stack.pop()
        
        if state[0] == target or state[1] == target:
            return path + [(state, "Goal reached!")]
        
        if depth >= max_depth:
            continue
        
        next_states = get_next_states(state, capacity_a, capacity_b)
        
        for next_state, rule in next_states:
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state, path + [(state, rule)], depth + 1))
    
    return None

def get_next_states(state, capacity_a, capacity_b):
    a, b = state
    next_states = []
    
    if a < capacity_a:
        next_states.append(((capacity_a, b), "Rule 1: Fill jug A completely"))
    
    if b < capacity_b:
        next_states.append(((a, capacity_b), "Rule 2: Fill jug B completely"))
    
    if a > 0:
        next_states.append(((0, b), "Rule 3: Empty jug A completely"))
    
    if b > 0:
        next_states.append(((a, 0), "Rule 4: Empty jug B completely"))
    
    if a > 0 and b < capacity_b:
        pour = min(a, capacity_b - b)
        next_states.append(((a - pour, b + pour), f"Rule 5: Pour {pour} units from jug A to jug B"))
    
    if b > 0 and a < capacity_a:
        pour = min(b, capacity_a - a)
        next_states.append(((a + pour, b - pour), f"Rule 6: Pour {pour} units from jug B to jug A"))
    
    return next_states

def print_solution(solution):
    if not solution:
        print("No solution found!")
        return
    
    print("\nSolution:")
    print("Initial state: (0, 0)")
    
    for i, (state, rule) in enumerate(solution):
        if i < len(solution) - 1:
            print(f"Apply: {rule}")
            print(f"State: {state}")
    
    print(f"Final state: {solution[-1][0]}")
    print(f"Total steps: {len(solution) - 1}")

def main():
    capacity_a = int(input("Enter capacity of Jug A: "))
    capacity_b = int(input("Enter capacity of Jug B: "))
    target = int(input("Enter target amount: "))
    
    print(f"\nWater Jug Problem: Jug A ({capacity_a}), Jug B ({capacity_b}), Target: {target}")
    
    solution = water_jug_dfs(capacity_a, capacity_b, target)
    print_solution(solution)

if __name__ == "__main__":
    main()