# profiler_metrics.py
from prometheus_client import Counter, Gauge

EXEC_TIME_GAUGE = Gauge("node_execution_time_seconds", "Execution time per node", ["node"])
TOTAL_FRAMES_COUNTER = Counter("node_total_frames", "Total frames per node", ["node"])
FILTERED_FRAMES_COUNTER = Counter("node_filtered_frames", "Filtered frames per node", ["node"])
GO_THROUGH_COUNTER = Counter("node_go_through_frames", "Passed frames per node", ["node"])
POSTPROCESS_BOXES_COUNTER = Counter(
    "postprocess_boxes_total", "Total number of boxes created during postprocessing", ["node"]
)
