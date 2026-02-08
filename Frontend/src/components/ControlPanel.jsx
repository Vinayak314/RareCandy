import React, { useState, useEffect } from 'react';
import { getGreeting } from '../api';

export default function ControlPanel({ banks, stocks, onSimulate, loading }) {
  const [mode, setMode] = useState('bank'); // 'bank' | 'stock'
  const [selectedBank, setSelectedBank] = useState('');
  const [bankShockPct, setBankShockPct] = useState(50);
  const [stockShocks, setStockShocks] = useState({});
  const [threshold, setThreshold] = useState(20);
  const [greeting, setGreeting] = useState('');

  useEffect(() => {
    getGreeting().then(data => setGreeting(data.greeting)).catch(() => { });
  }, []);

  const handleStockShockChange = (ticker, value) => {
    setStockShocks(prev => {
      const next = { ...prev };
      if (value === '' || Number(value) === 0) {
        delete next[ticker];
      } else {
        next[ticker] = Number(value);
      }
      return next;
    });
  };

  const handleRun = () => {
    if (mode === 'bank') {
      if (!selectedBank) return;
      onSimulate({ type: 'bank', bank: selectedBank, shock_pct: bankShockPct, failure_threshold: threshold });
    } else {
      const activeShocks = Object.fromEntries(
        Object.entries(stockShocks).filter(([, v]) => v > 0)
      );
      if (Object.keys(activeShocks).length === 0) return;
      onSimulate({ type: 'stock', shocks: activeShocks, failure_threshold: threshold });
    }
  };

  return (
    <>
      <div className="control-panel">
        <h2>Shock Controls</h2>

        {/* Mode Toggle */}
        <div className="mode-toggle">
          <button
            className={mode === 'bank' ? 'active' : ''}
            onClick={() => setMode('bank')}
          >
            Bank Shock
          </button>
          <button
            className={mode === 'stock' ? 'active' : ''}
            onClick={() => setMode('stock')}
          >
            Stock Shock
          </button>
        </div>

        {/* Bank Shock Mode */}
        {mode === 'bank' && (
          <div className="shock-config">
            <label>Target Bank</label>
            <select value={selectedBank} onChange={e => setSelectedBank(e.target.value)}>
              <option value="">Select bank...</option>
              {banks.map(b => (
                <option key={b.id} value={b.id}>
                  {b.id} — ${b.total_assets.toFixed(0)}B {b.margin_pct ? `(${b.margin_pct.toFixed(1)}% margin)` : ''}
                </option>
              ))}
            </select>

            {selectedBank && banks.find(b => b.id === selectedBank) && (
              <div className="selected-bank-info">
                {(() => {
                  const bank = banks.find(b => b.id === selectedBank);
                  return (
                    <>
                      <div className="bank-info-row">
                        <span>Health:</span>
                        <span className={`health-value ${bank.health > 60 ? 'good' : bank.health > 30 ? 'warn' : 'bad'}`}>
                          {bank.health?.toFixed(1) || 'N/A'}
                        </span>
                      </div>
                      <div className="bank-info-row">
                        <span>Margin Buffer:</span>
                        <span>${bank.margin_B?.toFixed(2) || '0'}B ({bank.margin_pct?.toFixed(1) || '0'}%)</span>
                      </div>
                    </>
                  );
                })()}
              </div>
            )}

            <label>Shock Severity: {bankShockPct}%</label>
            <input
              type="range" min={5} max={95} step={5}
              value={bankShockPct}
              onChange={e => setBankShockPct(Number(e.target.value))}
            />
            <div className="range-labels">
              <span>5%</span><span>50%</span><span>95%</span>
            </div>
          </div>
        )}

        {/* Stock Shock Mode */}
        {mode === 'stock' && (
          <div className="shock-config">
            <label>Devalue Stocks (set % drop)</label>
            <div className="stock-shock-list">
              {stocks.map(s => (
                <div key={s.ticker} className="stock-shock-row">
                  <span className="ticker">{s.ticker}</span>
                  <span className="price">${s.price.toFixed(2)}</span>
                  <input
                    type="number" min={0} max={95} step={5}
                    placeholder="0"
                    value={stockShocks[s.ticker] || ''}
                    onChange={e => handleStockShockChange(s.ticker, e.target.value)}
                  />
                  <span className="pct">%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Threshold */}
        <div className="threshold-config">
          <label>Failure Threshold: {threshold}</label>
          <input
            type="range" min={5} max={50} step={5}
            value={threshold}
            onChange={e => setThreshold(Number(e.target.value))}
          />
        </div>

        {/* Run Button */}
        <button className="run-btn" onClick={handleRun} disabled={loading}>
          {loading ? 'Simulating...' : 'Run Simulation'}
        </button>
      </div>

      {/* Greeting */}
      {greeting && (
        <div style={{ marginTop: '1rem', textAlign: 'center', fontSize: '0.9rem', color: '#444', fontStyle: 'italic', fontWeight: 'bold' }}>
          {greeting}
        </div>
      )}
    </>
  );
}
