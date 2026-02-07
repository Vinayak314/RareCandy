"""
Visualize a generated financial network with risk metrics from task.txt.
Run this script to see the scale-free topology with bank sizes and risk indicators.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from network_generator import NetworkGenerator
from cascade_simulator import CascadeSimulator


def visualize_network(n_banks: int = 20, m: int = 2, seed: int = 42, 
                      color_by: str = 'risk_score', show_vulnerable: bool = True):
    """
    Generate and visualize a financial network with CCP and risk metrics.
    
    Node size = Bank assets (bigger banks are bigger circles)
    Node color = Risk metric (configurable: risk_score, leverage, lcr, cds_spread, npl)
    CCP = Gold square in the center connected to all banks
    Edge thickness = Loan principal
    Red border = Vulnerable banks (high contagion risk)
    Triangle marker = Banks with CDS warning signal
    
    Args:
        n_banks: Number of banks in network
        m: Barabasi-Albert connectivity parameter
        seed: Random seed for reproducibility
        color_by: Metric to color nodes by ('risk_score', 'leverage', 'lcr', 'cds_spread', 'npl', 'volatility')
        show_vulnerable: Highlight vulnerable banks with red borders
    """
    # Generate network
    gen = NetworkGenerator(seed=seed)
    G, banks, loans, ccp = gen.generate_full_network(n_banks, m)
    
    # Create cascade simulator for risk analysis
    cs = CascadeSimulator(banks, loans, ccp)
    
    # Get risk assessment
    risk_assessment = gen.get_risk_assessment(banks, loans)
    systemic_report = cs.get_systemic_risk_report()
    vulnerable_banks = cs.identify_vulnerable_banks(threshold=0.5)
    
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
    
    # Node colors based on selected metric
    color_values = []
    cmap = plt.cm.RdYlGn_r  # Red=high risk, Green=low risk
    vmin, vmax = 0, 1
    color_label = 'Risk Score'
    
    for n in bank_nodes:
        bank = banks[n]
        if color_by == 'risk_score':
            color_values.append(bank.calculate_counterparty_risk_score())
            color_label = 'Counterparty Risk Score'
        elif color_by == 'leverage':
            lev = bank.calculate_leverage()
            color_values.append(min(lev / 30.0, 1.0))  # Normalize to 0-1, cap at 30x
            color_label = 'Leverage (normalized)'
        elif color_by == 'lcr':
            lcr = bank.calculate_lcr()
            # Invert: low LCR = high risk = red
            color_values.append(max(0, min(1.0, 1.5 - lcr)) if lcr != float('inf') else 0)
            color_label = 'LCR Risk (inverted)'
        elif color_by == 'cds_spread':
            color_values.append(min(bank.cds_spread / 300.0, 1.0))
            color_label = 'CDS Spread (normalized)'
        elif color_by == 'npl':
            color_values.append(min(bank.calculate_npl_ratio() / 0.1, 1.0))
            color_label = 'NPL Ratio (normalized)'
        elif color_by == 'volatility':
            color_values.append(min(bank.stock_volatility / 0.1, 1.0))
            color_label = 'Stock Volatility (normalized)'
        else:
            color_values.append(bank.risk_weight)
            color_label = 'Systemic Risk Weight'
    
    # Identify special banks
    cds_warning_nodes = [n for n in bank_nodes if banks[n].is_cds_warning()]
    lcr_desperate_nodes = [n for n in bank_nodes if banks[n].is_lcr_desperate()]
    high_leverage_nodes = [n for n in bank_nodes if banks[n].is_leverage_high_risk()]
    vulnerable_nodes = [n for n in bank_nodes if banks[n].bank_id in vulnerable_banks]
    
    # Edge widths based on loan principal
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        if u == -1 or v == -1:
            edge_widths.append(1.0)
            edge_colors.append('gold')
        else:
            matching = [l for l in loans 
                       if (l.lender_id == banks[u].bank_id and l.borrower_id == banks[v].bank_id)
                       or (l.lender_id == banks[v].bank_id and l.borrower_id == banks[u].bank_id)]
            if matching:
                edge_widths.append(0.5 + 2 * matching[0].principal / 1000)
            else:
                edge_widths.append(1)
            edge_colors.append('gray')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ===== LEFT PLOT: Network Visualization =====
    ax = axes[0]
    
    # Draw bank-to-bank edges
    bank_edges = [(u, v) for u, v in G.edges() if u != -1 and v != -1]
    bank_edge_widths = [edge_widths[i] for i, (u, v) in enumerate(G.edges()) if u != -1 and v != -1]
    nx.draw_networkx_edges(G, pos, edgelist=bank_edges, ax=ax, width=bank_edge_widths, alpha=0.5, edge_color='gray')
    
    # Draw CCP edges (gold, dashed)
    ccp_edges = [(u, v) for u, v in G.edges() if u == -1 or v == -1]
    nx.draw_networkx_edges(G, pos, edgelist=ccp_edges, ax=ax, width=1.5, alpha=0.7, edge_color='gold', style='dashed')
    
    # Draw regular bank nodes
    regular_nodes = [n for n in bank_nodes if n not in vulnerable_nodes]
    regular_sizes = [bank_sizes[bank_nodes.index(n)] for n in regular_nodes]
    regular_colors = [color_values[bank_nodes.index(n)] for n in regular_nodes]
    
    if regular_nodes:
        nodes_regular = nx.draw_networkx_nodes(
            G, pos, nodelist=regular_nodes, ax=ax,
            node_size=regular_sizes,
            node_color=regular_colors,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolors='black',
            linewidths=1
        )
    
    # Draw vulnerable bank nodes with red border
    if show_vulnerable and vulnerable_nodes:
        vuln_sizes = [bank_sizes[bank_nodes.index(n)] for n in vulnerable_nodes]
        vuln_colors = [color_values[bank_nodes.index(n)] for n in vulnerable_nodes]
        nodes_vuln = nx.draw_networkx_nodes(
            G, pos, nodelist=vulnerable_nodes, ax=ax,
            node_size=vuln_sizes,
            node_color=vuln_colors,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolors='red',
            linewidths=3
        )
    
    # Add warning markers for CDS warning banks
    if cds_warning_nodes:
        for n in cds_warning_nodes:
            x, y = pos[n]
            ax.plot(x, y + 0.12, marker='^', color='red', markersize=12, 
                   markeredgecolor='darkred', markeredgewidth=1)
    
    # Add markers for LCR desperate banks
    if lcr_desperate_nodes:
        for n in lcr_desperate_nodes:
            x, y = pos[n]
            ax.plot(x, y - 0.12, marker='v', color='orange', markersize=10,
                   markeredgecolor='darkorange', markeredgewidth=1)
    
    # Draw CCP node (large gold square)
    nx.draw_networkx_nodes(
        G, pos, nodelist=[-1], ax=ax,
        node_size=1500,
        node_color='gold',
        node_shape='s',
        edgecolors='black',
        linewidths=2
    )
    
    # Labels
    labels = {n: f"B{n}" for n in bank_nodes}
    labels[-1] = "CCP"
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(color_label, fontsize=10)
    
    ax.set_title(f"Financial Network ({n_banks} Banks, {len(loans)} Loans)\nColored by: {color_label}", fontsize=12)
    ax.axis('off')
    
    # Legend for markers
    legend_elements = [
        mpatches.Patch(facecolor='gold', edgecolor='black', label='CCP'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markeredgecolor='red', markeredgewidth=2, markersize=10, label='Vulnerable Bank'),
        plt.Line2D([0], [0], marker='^', color='red', markersize=10, linestyle='None', label='CDS Warning (≥200bps)'),
        plt.Line2D([0], [0], marker='v', color='orange', markersize=10, linestyle='None', label='LCR Desperate (<100%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
    
    # ===== RIGHT PLOT: Risk Metrics Summary =====
    ax2 = axes[1]
    ax2.axis('off')
    
    # Create summary text
    summary = systemic_report['network_summary']
    risk_summary = risk_assessment['summary']
    
    text = f"""
NETWORK RISK SUMMARY (task.txt Metrics)
{'='*50}

A. BALANCE SHEET FUNDAMENTALS
{'─'*50}
Average Leverage Ratio:     {risk_summary['avg_leverage']:.2f}x
Max Leverage Ratio:         {risk_summary['max_leverage']:.2f}x
High Leverage Banks (>20x): {risk_summary['high_leverage_banks']}

Average LCR:                {risk_summary['avg_lcr']:.2%}
Min LCR:                    {risk_summary['min_lcr']:.2%}
Desperate Banks (LCR<100%): {summary['desperate_banks_lcr']}

Average NPL Ratio:          {risk_summary['avg_npl_ratio']:.2%}
Max NPL Ratio:              {risk_summary['max_npl_ratio']:.2%}

B. MARKET SIGNALS
{'─'*50}
Average CDS Spread:         {risk_summary['avg_cds_spread']:.1f} bps
Max CDS Spread:             {risk_summary['max_cds_spread']:.1f} bps
CDS Warning Banks (≥200):   {summary['cds_warning_banks']}

Average Stock Volatility:   {risk_summary['avg_volatility']:.2%}

C. NETWORK POSITION
{'─'*50}
Total Interbank Exposure:   ${risk_summary['total_interbank_exposure']:,.0f}
Average Asset Correlation:  {risk_summary['avg_asset_correlation']:.3f}

CONTAGION RISK ANALYSIS
{'─'*50}
Average Contagion Risk:     {summary['avg_contagion_risk']:.3f}
Max Contagion Risk:         {summary['max_contagion_risk']:.3f}
High Risk Banks:            {summary['high_risk_banks']}
Medium Risk Banks:          {summary['medium_risk_banks']}
Low Risk Banks:             {summary['low_risk_banks']}

TOP 5 VULNERABLE BANKS
{'─'*50}
{', '.join(systemic_report['top_5_vulnerable_banks'][:5])}

CCP STATUS
{'─'*50}
CCP Equity:                 ${ccp.equity:,.0f}
CCP Default Fund:           ${ccp.default_fund:,.0f}
"""
    
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('network_risk_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nNetwork saved to 'network_risk_visualization.png'")
    print(f"Nodes: {n_banks} banks + 1 CCP, Edges: {len(loans)} loans")
    print(f"Vulnerable banks: {len(vulnerable_banks)}")
    print(f"Hub bank (most connections): B{max([(n, G.degree(n)) for n in bank_nodes], key=lambda x: x[1])[0]}")


def visualize_risk_comparison(n_banks: int = 20, m: int = 2, seed: int = 42):
    """
    Generate multiple visualizations comparing different risk metrics.
    Creates a 2x3 grid showing the network colored by different metrics.
    """
    gen = NetworkGenerator(seed=seed)
    G, banks, loans, ccp = gen.generate_full_network(n_banks, m)
    cs = CascadeSimulator(banks, loans, ccp)
    vulnerable_banks = cs.identify_vulnerable_banks(threshold=0.5)
    
    # Add CCP node
    G.add_node(-1)
    for bank_node in banks.keys():
        G.add_edge(-1, bank_node)
    
    pos = nx.spring_layout(G, seed=seed, k=2, fixed=[-1], pos={-1: (0, 0)})
    bank_nodes = list(banks.keys())
    
    # Node sizes
    max_assets = max(b.assets for b in banks.values())
    bank_sizes = [200 + 500 * (banks[n].assets / max_assets) for n in bank_nodes]
    
    # Metrics to visualize
    metrics = [
        ('risk_score', 'Counterparty Risk Score', lambda b: b.calculate_counterparty_risk_score()),
        ('leverage', 'Leverage Ratio', lambda b: min(b.calculate_leverage() / 30.0, 1.0)),
        ('lcr', 'LCR Risk', lambda b: max(0, min(1.0, 1.5 - b.calculate_lcr())) if b.calculate_lcr() != float('inf') else 0),
        ('cds_spread', 'CDS Spread', lambda b: min(b.cds_spread / 300.0, 1.0)),
        ('npl', 'NPL Ratio', lambda b: min(b.calculate_npl_ratio() / 0.1, 1.0)),
        ('volatility', 'Stock Volatility', lambda b: min(b.stock_volatility / 0.1, 1.0)),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric_name, label, metric_func) in enumerate(metrics):
        ax = axes[idx]
        
        color_values = [metric_func(banks[n]) for n in bank_nodes]
        
        # Draw edges
        bank_edges = [(u, v) for u, v in G.edges() if u != -1 and v != -1]
        nx.draw_networkx_edges(G, pos, edgelist=bank_edges, ax=ax, width=0.5, alpha=0.3, edge_color='gray')
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos, nodelist=bank_nodes, ax=ax,
            node_size=bank_sizes,
            node_color=color_values,
            cmap=plt.cm.RdYlGn_r,
            vmin=0, vmax=1,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Draw CCP
        nx.draw_networkx_nodes(G, pos, nodelist=[-1], ax=ax, node_size=800,
                              node_color='gold', node_shape='s', edgecolors='black', linewidths=1)
        
        # Labels
        labels = {n: f"{n}" for n in bank_nodes}
        labels[-1] = "CCP"
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6)
    
    plt.suptitle(f'Financial Network Risk Metrics Comparison ({n_banks} Banks)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('network_risk_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison saved to 'network_risk_comparison.png'")


if __name__ == '__main__':
    print("Generating network visualization with risk metrics...")
    visualize_network(n_banks=25, m=2, color_by='risk_score', show_vulnerable=True)
    
    print("\nGenerating risk comparison visualization...")
    visualize_risk_comparison(n_banks=25, m=2)
