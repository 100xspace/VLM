QKV computation: O(3nd²) - Linear projections
Attention scores: O(n²d) - Matrix multiplication QKT
Attention weights: O(n²) - Softmax computation
Weighted values: O(n²d) - Multiply attention weights with V
Output projection: O(nd²) - Final linear layer
