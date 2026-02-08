export const mockGraphData = {
  nodes: [
    { id: "CCP", name: "Central Counterparty", group: "CCP", val: 20 },
    { id: "Bank_A", name: "Bank A (Tier 1)", group: "Bank", val: 10 },
    { id: "Bank_B", name: "Bank B (Tier 1)", group: "Bank", val: 10 },
    { id: "Bank_C", name: "Bank C (Tier 2)", group: "Bank", val: 5 },
    { id: "Bank_D", name: "Bank D (Tier 2)", group: "Bank", val: 5 },
    { id: "Bank_E", name: "Bank E (Tier 3)", group: "Bank", val: 3 },
    { id: "Bank_F", name: "Bank F (Tier 3)", group: "Bank", val: 3 },
    { id: "Bank_G", name: "Bank G (Tier 3)", group: "Bank", val: 3 },
  ],
  links: [
    // CCP connections (clearing members)
    { source: "Bank_A", target: "CCP", value: 5 },
    { source: "Bank_B", target: "CCP", value: 5 },
    { source: "Bank_C", target: "CCP", value: 2 },
    { source: "Bank_D", target: "CCP", value: 2 },
    
    // Interbank exposures (could represent credit lines or bilateral trades)
    { source: "Bank_A", target: "Bank_B", value: 3 },
    { source: "Bank_A", target: "Bank_C", value: 1 },
    { source: "Bank_B", target: "Bank_D", value: 1 },
    { source: "Bank_C", target: "Bank_E", value: 1 },
    { source: "Bank_D", target: "Bank_F", value: 1 },
    { source: "Bank_E", target: "Bank_G", value: 1 },
  ]
};
