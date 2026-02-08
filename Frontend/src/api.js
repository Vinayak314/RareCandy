const API_BASE = '/api';

async function fetchJSON(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export function getBanks() {
  return fetchJSON('/banks');
}

export function getNetwork() {
  return fetchJSON('/network');
}

export function getStocks() {
  return fetchJSON('/stocks');
}

export function getHoldings() {
  return fetchJSON('/holdings');
}

export function simulateBankShock(bank, shockPct, failureThreshold = 20) {
  return fetchJSON('/simulate/bank-shock', {
    method: 'POST',
    body: JSON.stringify({ bank, shock_pct: shockPct, failure_threshold: failureThreshold }),
  });
}

export function simulateStockShock(shocks, failureThreshold = 20) {
  return fetchJSON('/simulate/stock-shock', {
    method: 'POST',
    body: JSON.stringify({ shocks, failure_threshold: failureThreshold }),
  });
}

export function resetSimulation(useMlMargins = true) {
  return fetchJSON('/simulate/reset', { 
    method: 'POST',
    body: JSON.stringify({ use_ml_margins: useMlMargins }),
  });
}

export function getGreeting() {
  return fetchJSON('/greeting');
}

// ─── ML Model API Functions ─────────────────────────────────────────────────

export function getMlStatus() {
  return fetchJSON('/ml/status');
}

export function getMarginRequirements() {
  return fetchJSON('/ml/margins');
}

export function getBankMarginDetails(bankName) {
  return fetchJSON(`/ml/margins/${bankName}`);
}

export function regenerateMargins(useMl = true) {
  return fetchJSON('/ml/regenerate-margins', {
    method: 'POST',
    body: JSON.stringify({ use_ml: useMl }),
  });
}

export function simulateWithCustomMargins(config) {
  return fetchJSON('/ml/simulate-with-margins', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}
