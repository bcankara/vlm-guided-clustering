# Experiment Report: vlm_hierarchical

**Algorithm:** hierarchical  
**Model:** gemini-3-flash-preview  
**Started:** 2025-12-29 00:48:21  

---

## Processing Steps

| Step | Cluster | Points | Action | Result |
|:----:|---------|-------:|:------:|--------|
| 1 | **C0** | 7,316 | ✂️ | C0.1(3450), C0.2(2553), C0.3(1313) |
| 2 | **C1** | 2,684 | ✂️ | C1.1(1452), C1.2(1232) |
| 3 | **C0.1** | 3,450 | ✂️ | C0.1.1(1401), C0.1.2(1154), C0.1.3(895) |
| 4 | **C0.2** | 2,553 | ✂️ | C0.2.1(934), C0.2.2(917), C0.2.3(702) |
| 5 | **C0.3** | 1,313 | ✂️ | C0.3.1(642), C0.3.2(671) |
| 6 | **C1.1** | 1,452 | ❄️ | FROZEN |
| 7 | **C1.2** | 1,232 | ❄️ | FROZEN |
| 8 | **C0.1.1** | 1,401 | ✂️ | C0.1.1.1(1055), C0.1.1.2(346) |
| 9 | **C0.1.2** | 1,154 | ❄️ | FROZEN |
| 10 | **C0.1.3** | 895 | ❄️ | FROZEN |
| 11 | **C0.2.1** | 934 | ❄️ | FROZEN |
| 12 | **C0.2.2** | 917 | ❄️ | FROZEN |
| 13 | **C0.2.3** | 702 | ❄️ | FROZEN |
| 14 | **C0.3.1** | 642 | ❄️ | FROZEN |
| 15 | **C0.3.2** | 671 | ❄️ | FROZEN |
| 16 | **C0.1.1.1** | 1,055 | ❄️ | FROZEN |
| 17 | **C0.1.1.2** | 346 | ❄️ | FROZEN |

---

## Merge Phase

- **Pre-merge Clusters:** 11
- **Post-merge Clusters:** 4
- **Pre-merge ARI:** 0.4470
- **Post-merge ARI:** 0.5697

**Merge Groups:**
- Group 0: ['C0.2.1', 'C0.1.3', 'C0.3.2']
- Group 1: ['C0.2.3', 'C0.1.1.2']
- Group 2: ['C1.1', 'C0.2.2', 'C0.1.2', 'C0.1.1.1']
- Group 3: ['C1.2', 'C0.3.1']

---

## Final Results

- **Final Clusters:** 4
- **ARI:** 0.5697
- **NMI:** 0.7148
- **Total Steps:** 18
- **Duration:** 214.4s
