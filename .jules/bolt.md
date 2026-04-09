## 2024-04-09 - Add LSH bucketing to NearDedupStage
/ Learning: NearDedupStage was previously O(n^2), calculating exact MinHash similarities across all candidates, slowing down at large record counts. LSH bucketing reduces candidate pool efficiently.
/ Action: In the future, keep looking for O(n^2) scaling issues with processing lists of records directly in Python memory.
