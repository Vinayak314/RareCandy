import React from 'react';

export default function SimulationResults({ result }) {
  if (!result) return null;

  const {
    num_failed, total_banks, collapse_ratio, total_asset_loss,
    avg_survivor_health, rounds, system_collapsed, failed_banks,
    contagion_history, banks, ccp_payoff_B, payoff_breakdown
  } = result;


  // Sort banks by asset loss descending
  const sortedBanks = [...(banks || [])].sort((a, b) => b.asset_loss - a.asset_loss);
  const topAffected = sortedBanks.filter(b => b.asset_loss > 0).slice(0, 10);

  return (
    <div className="simulation-results">
      <h2>Simulation Results</h2>

      {/* Summary Cards */}
      <div className="result-cards">
        <div className={`card ${system_collapsed ? 'danger' : num_failed > 0 ? 'warning' : 'safe'}`}>
          <div className="card-value">{num_failed}/{total_banks}</div>
          <div className="card-label">Banks Failed</div>
        </div>
        <div className="card">
          <div className="card-value">${Number(total_asset_loss || 0).toFixed(1)}B</div>
          <div className="card-label">Total Asset Loss</div>
        </div>
        <div className="card">
          <div className="card-value">{Number(avg_survivor_health || 0).toFixed(1)}</div>
          <div className="card-label">Avg Survivor Health</div>
        </div>
        <div className="card">
          <div className="card-value">{rounds}</div>
          <div className="card-label">Rounds to Stability</div>
        </div>
        {ccp_payoff_B !== undefined && ccp_payoff_B !== null && (
          <div className={`card ccp-card ${ccp_payoff_B > 0 ? 'warning' : 'safe'}`}>
            <div className="card-value">${Number(ccp_payoff_B).toFixed(1)}B</div>
            <div className="card-label">CCP Payoff</div>
          </div>
        )}
      </div>

      {system_collapsed && (
        <div className="collapse-banner">SYSTEMIC COLLAPSE DETECTED</div>
      )}

      {/* Failed Banks */}
      {failed_banks && failed_banks.length > 0 && (
        <div className="failed-section">
          <h3>Failed Banks</h3>
          <div className="failed-tags">
            {failed_banks.map(b => (
              <span key={b} className="failed-tag">{b}</span>
            ))}
          </div>
        </div>
      )}

      {/* Top Affected Banks Table */}
      {topAffected.length > 0 && (
        <div className="affected-table">
          <h3>Most Affected Banks</h3>
          <table>
            <thead>
              <tr>
                <th>Bank</th>
                <th>Health</th>
                <th>Asset Loss</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {topAffected.map(b => (
                <tr key={b.id} className={b.failed ? 'row-failed' : ''}>
                  <td><strong>{b.id}</strong></td>
                  <td>
                    <div className="health-bar-wrapper">
                      <div
                        className="health-bar"
                        style={{
                          width: `${b.health}%`,
                          backgroundColor: b.health > 50 ? '#4CAF50' : b.health > 20 ? '#FF9800' : '#D91A25',
                        }}
                      />
                      <span>{b.health.toFixed(1)}</span>
                    </div>
                  </td>
                  <td>${Number(b.asset_loss || 0).toFixed(2)}B</td>
                  <td>{b.failed ? 'FAILED' : 'OK'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Contagion Progression */}
      {contagion_history && contagion_history.length > 0 && (
        <div className="contagion-timeline">
          <h3>Contagion Progression</h3>
          {contagion_history.map(h => (
            <div key={h.round} className="timeline-step">
              <span className="round-num">R{h.round}</span>
              <span className="round-info">
                {h.failed_banks} failed (+{h.newly_failed.length} new) — Loss: ${h.total_asset_loss.toFixed(1)}B
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
