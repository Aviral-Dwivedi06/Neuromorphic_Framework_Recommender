from models.recommender import NeuromorphicModel
from visualizer import plot_comparison
import sys

# 1. Initialize and Train
recommender = NeuromorphicModel()
try:
    recommender.train('data/neuromorphic_dataset.csv')
except FileNotFoundError:
    print("Error: 'data/neuromorphic_dataset.csv' not found.")
    sys.exit()

print("\n" + "="*50)
print("   NEUROMORPHIC SYSTEM RECOMMENDER - TEAM MORPHEUS")
print("="*50)
print("INSTRUCTIONS: Enter your constraints below.")
print(">> TO FINISH AND VIEW COMPARISON GRAPH: Type 'exit' at any prompt.")

history = []

while True:
    try:
        print("\n" + "-"*30)
        u_node = input("Process Node (nm) [or 'exit' to finish and view comparison graph]: ").strip()
        if u_node.lower() == 'exit': break
        
        u_power = input("Power Budget (mW) [or 'exit' to finish and view comparison graph]: ").strip()
        if u_power.lower() == 'exit': break
        
        print("Apps: High-Perf CNN, Character Recog, Edge Computing, SNN Accelerator, Ultra-Low Edge")
        u_app = input("Selection [or 'exit']: ").strip()
        if u_app.lower() == 'exit': break

        # Process Inputs
        node = float(u_node)
        power = float(u_power)

        # 2. Get Prediction
        rec = recommender.recommend(node, power, u_app)

        # 3. Output Results
        print(f"\n[RECOMMENDATION FOR {u_app.upper()}]")
        for key, value in rec.items():
            if key == "Rationale" and "⚠️" in value:
                # Highlighting warning in terminal
                print(f"{key:15}: >>> {value} <<<")
            else:
                print(f"{key:15}: {value}")
        
        # Log to history for visualizer
        history.append({'app': u_app, 'power': power, 'node': node})

    except ValueError:
        print("\nInvalid input. Please enter numbers for Node and Power.")
    except Exception as e:
        print(f"\nNotice: {e}")

# 4. Final Transition to Visualizer
if history:
    print("\n" + "="*50)
    print("Exiting session... Generating Power-Efficiency Chart...")
    print("="*50)
    plot_comparison(history)
else:
    print("\nNo data collected. Exiting.")