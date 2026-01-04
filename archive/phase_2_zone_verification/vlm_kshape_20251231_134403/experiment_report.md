# Experiment Report: vlm_kshape

**Algorithm:** kshape  
**Model:** gemini-3-pro-preview  
**Started:** 2025-12-31 13:44:03  

---

## Processing Steps

| Step | Cluster | Points | Action | Result |
|:----:|---------|-------:|:------:|--------|
| 1 | **C0** | 5,023 | ✂️ | C0.1(2523), C0.2(2500) |
| 2 | **C1** | 4,977 | ❄️ | FROZEN |
| 3 | **C0.1** | 2,523 | ❄️ | FROZEN |
| 4 | **C0.2** | 2,500 | ❄️ | FROZEN |

---

## Merge Phase

- **Pre-merge Clusters:** 3
- **Post-merge Clusters:** 3
- **Pre-merge ARI:** 0.7112
- **Post-merge ARI:** 0.7112

**Merge Groups:**
- Group 0: ['C0.1']
- Group 1: ['C0.2']
- Group 2: ['C1']

---

## Final Results

- **Final Clusters:** 3
- **ARI:** 0.7112
- **NMI:** 0.8471
- **Total Steps:** 5
- **Duration:** 1023.5s
