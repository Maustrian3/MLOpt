import csv

SGP_INSTANCES = [
    # ---------------------------------------------------------
    # TIER 1: EASY (Group Size 3)
    # ---------------------------------------------------------
    (3, 9, 2),   
    (3, 9, 3),   
    (3, 9, 4),   
    (4, 12, 2),  
    (4, 12, 3),  
    
    # ---------------------------------------------------------
    # TIER 2: MODERATE 
    # ---------------------------------------------------------
    (4, 12, 4),  
    (4, 12, 5),  
    (4, 16, 2),  
    (4, 16, 3),  
    (4, 16, 4),  
    (4, 16, 5),  
    (5, 15, 3),  
    (5, 15, 4),  
    
    # ---------------------------------------------------------
    # TIER 3: HARD
    # ---------------------------------------------------------
    (5, 15, 5),  
    (5, 15, 6),  
    (5, 20, 3),  
    (5, 20, 4),  
    (5, 20, 5),  
    (6, 18, 3),  
    (6, 18, 4)   
]

def generate_instances_csv():
    filename = "sgp_solvable_instances.csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the column headers
        writer.writerow(["num_groups", "num_players", "num_weeks"])
        
        # Write the list of tuples
        writer.writerows(SGP_INSTANCES)
        
    print(f"Successfully saved {len(SGP_INSTANCES)} instances to '{filename}'.")

if __name__ == "__main__":
    generate_instances_csv()