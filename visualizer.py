import matplotlib.pyplot as plt

def plot_comparison(results_list):
    """
    results_list: List of dictionaries containing {label, power, tech}
    """
    apps = [r['app'] for r in results_list]
    powers = [r['power'] for r in results_list]
    nodes = [r['node'] for r in results_list]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for Power (Log scale since power varies from nW to Watts)
    color = 'tab:blue'
    ax1.set_xlabel('Application Scenario')
    ax1.set_ylabel('Power Budget (mW) - Log Scale', color=color)
    ax1.bar(apps, powers, color=color, alpha=0.6, label='Power (mW)')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color)

    # Line chart for Tech Node
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tech Node (nm)', color=color)
    ax2.plot(apps, nodes, color=color, marker='o', linewidth=2, label='Node (nm)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Neuromorphic Recommendation Comparison')
    fig.tight_layout()
    plt.show()