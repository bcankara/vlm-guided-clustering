# Experiment Report: vlm_kshape

**Algorithm:** kshape  
**Model:** gemini-3-flash-preview  
**Started:** 2025-12-29 21:17:13  

---

## Processing Steps

| Step | Cluster | Points | Action | Result |
|:----:|---------|-------:|:------:|--------|
| 1 | **C0** | 5,023 | ✂️ | C0.1(1309), C0.2(1191), C0.3(2523) |
| 2 | **C1** | 4,977 | ✂️ | C1.1(2477), C1.2(2500) |
| 3 | **C0.1** | 1,309 | ❄️ | FROZEN |
| 4 | **C0.2** | 1,191 | ❄️ | FROZEN |
| 5 | **C0.3** | 2,523 | ✂️ | C0.3.1(864), C0.3.2(1659) |
| 6 | **C1.1** | 2,477 | ❄️ | FROZEN |
| 7 | **C1.2** | 2,500 | ❄️ | FROZEN |
| 8 | **C0.3.1** | 864 | ❄️ | FROZEN |
| 9 | **C0.3.2** | 1,659 | ❄️ | FROZEN |

---

## Merge Phase

- **Pre-merge Clusters:** 6
- **Post-merge Clusters:** 2
- **Pre-merge ARI:** 0.8243
- **Post-merge ARI:** 0.3319

**Merge Groups:**
- Group 0: ['C0.3.1', 'C0.3.2']
- Group 1: ['C1.2', 'C1.1', 'C0.1', 'C0.2']

---

## Final Results

- **Final Clusters:** 6
- **ARI:** 0.8243
- **NMI:** 0.8853
- **Total Steps:** 10
- **Duration:** 1047.9s
