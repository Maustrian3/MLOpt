# Algorithmic selection
## Features
### 1. Number of golfers `num_golfers`
### 2. Number of Weeks `num_weeks`
### 3. Group size `group_size`
We chose group size instead of number of groups, as seemed more relevant to us.
### 4. Tightness index `T` (pairing saturation)

**Definition:**  
Each golfer meets `group_size − 1` new opponents per week.  
The maximum number of distinct opponents per golfer is `num_golfers − 1`.  
Define:

T = num_weeks * (group_size - 1) / (num_golfers - 1)

**Interpretation:**  
- `T` close to 1 → golfers must almost exhaust all possible opponents → **very tight** instance.  
- `T` much smaller than 1 → looser constraints, more flexibility.

### TODO Mention some combinatoric
Reference: ChatGPT

### 5. Local search gradient probe `ls_improv_rate`
**Definition:**  
Run a local search (e.g., pair swap) for *5–20 iterations* starting from a random solution.  
Measure how much the conflict count improves.

`ls_improv_rate = (conflicts_initial - conflicts_after) / iterations`

**Interpretation:**  
- High improvement → the search landscape is smooth → metaheuristics are likely to perform well.  
- Low or zero improvement → the search landscape is flat → exact solvers may handle this structure better.