## 2024-05-23 - [Executor Overhead for Lightweight Tasks]
**Learning:** Offloading lightweight tasks (like appending to a deque) to `loop.run_in_executor` introduces significant overhead that outweighs the execution cost.
**Action:** Always measure the cost of the task vs the overhead of dispatch. Only offload heavy CPU-bound tasks (like neural network inference), keep lightweight state management on the main loop.
