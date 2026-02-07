"""
Visualize a generated financial network.
Run this script to see the scale-free topology with bank sizes.
"""

import matplotlib.pyplot as plt
import networkx as nx
from network_generator import NetworkGenerator


def visualize_network(n_banks: int = 20, m: int = 2, seed: int = 42):
    """
    Generate and visualize a financial network with CCP.
    
    Node size = Bank assets (bigger banks are bigger circles)
    Node color = Risk weight (redder = more systemically important)
    CCP = Gold star in the center connected to all banks
    Edge thickness = Loan principal
    """
    # Generate
    gen = NetworkGenerator(seed=seed)
    G, banks, loans, ccp = gen.generate_full_network(n_banks, m)
    
    # Add CCP node to graph (node ID = -1 to distinguish)
    G.add_node(-1)
    for bank_node in banks.keys():
        G.add_edge(-1, bank_node)
    
    # Layout (spring layout with CCP fixed at center)
    pos = nx.spring_layout(G, seed=seed, k=2, fixed=[-1], pos={-1: (0, 0)})
    
    # Separate bank nodes from CCP
    bank_nodes = list(banks.keys())
    
    # Node sizes based on assets (scaled for visibility)
    max_assets = max(b.assets for b in banks.values())
    bank_sizes = [300 + 700 * (banks[n].assets / max_assets) for n in bank_nodes]
    
    # Node colors based on risk weight (0=green, 1=red)
    bank_colors = [banks[n].risk_weight for n in bank_nodes]
    
    # Edge widths based on loan principal (only for bank-to-bank edges)
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        if u == -1 or v == -1:
            # CCP edges (dashed gold lines)
            edge_widths.append(1.0)
            edge_colors.append('gold')
        else:
            # Bank-to-bank loans
            matching = [l for l in loans 
                       if (l.lender_id == banks[u].bank_id and l.borrower_id == banks[v].bank_id)
                       or (l.lender_id == banks[v].bank_id and l.borrower_id == banks[u].bank_id)]
            if matching:
                edge_widths.append(0.5 + 2 * matching[0].principal / 1000)
            else:
                edge_widths.append(1)
            edge_colors.append('gray')
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Draw bank-to-bank edges (gray)
    bank_edges = [(u, v) for u, v in G.edges() if u != -1 and v != -1]
    bank_edge_widths = [edge_widths[i] for i, (u, v) in enumerate(G.edges()) if u != -1 and v != -1]
    nx.draw_networkx_edges(G, pos, edgelist=bank_edges, ax=ax, width=bank_edge_widths, alpha=0.5, edge_color='gray')
    
    # Draw CCP edges (gold, dashed)
    ccp_edges = [(u, v) for u, v in G.edges() if u == -1 or v == -1]
    nx.draw_networkx_edges(G, pos, edgelist=ccp_edges, ax=ax, width=1.5, alpha=0.7, edge_color='gold', style='dashed')
    
    # Draw bank nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, nodelist=bank_nodes, ax=ax,
        node_size=bank_sizes,
        node_color=bank_colors,
        cmap=plt.cm.RdYlGn_r,
        vmin=0, vmax=1
    )
    
    # Draw CCP node (large gold star)
    nx.draw_networkx_nodes(
        G, pos, nodelist=[-1], ax=ax,
        node_size=1500,
        node_color='gold',
        node_shape='s',  # Square for CCP
        edgecolors='black',
        linewidths=2
    )
    
    # Labels
    labels = {n: f"B{n}" for n in bank_nodes}
    labels[-1] = "CCP"
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')
    
    # Colorbar
    plt.colorbar(nodes, ax=ax, label='Systemic Risk Weight')
    
    # Title and info
    ax.set_title(f"Financial Network with CCP ({n_banks} Banks, {len(loans)} Loans)", fontsize=14)
    ax.axis('off')
    
    # Stats box
    total_assets = sum(b.assets for b in banks.values())
    total_loans = sum(l.principal for l in loans)
    stats_text = f"Total Assets: ${total_assets:,.0f}\nTotal Loans: ${total_loans:,.0f}\nCCP Equity: ${ccp.equity:,.0f}\nCCP Default Fund: ${ccp.default_fund:,.0f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('network_visualization.png', dpi=150)
    plt.show()
    
    print(f"\nNetwork saved to 'network_visualization.png'")
    print(f"Nodes: {n_banks} banks + 1 CCP, Edges: {len(loans)} loans")
    print(f"Hub bank (most connections): B{max([(n, G.degree(n)) for n in bank_nodes], key=lambda x: x[1])[0]}")


if __name__ == '__main__':
    visualize_network(n_banks=25, m=2)
