"""
Qwen Debug Controller Node
Comprehensive debugging and performance analysis for Qwen nodes
"""

import logging
import torch
import psutil
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import traceback
import sys

# Import existing debug infrastructure
try:
    from . import debug_patch
    from .qwen_logger import QwenLogger
except ImportError:
    import debug_patch
    from qwen_logger import QwenLogger


class DebugCollector:
    """Collects and manages debug information from across the system"""

    def __init__(self):
        self.execution_logs = []
        self.performance_data = defaultdict(list)
        self.memory_snapshots = []
        self.error_log = []
        self.node_statistics = defaultdict(lambda: {
            'executions': 0,
            'total_time': 0.0,
            'errors': 0,
            'avg_memory': 0.0
        })
        self.start_time = time.time()

    def log_execution(self, component: str, message: str, level: str = "INFO"):
        """Log an execution event"""
        self.execution_logs.append({
            'timestamp': datetime.now().isoformat(),
            'elapsed': time.time() - self.start_time,
            'component': component,
            'level': level,
            'message': message
        })

    def log_performance(self, component: str, duration: float, details: Dict = None):
        """Log performance metrics"""
        self.performance_data[component].append({
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'details': details or {}
        })

    def log_memory_snapshot(self):
        """Capture current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()

        self.memory_snapshots.append({
            'timestamp': datetime.now().isoformat(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        })

    def log_error(self, component: str, error: Exception, context: Dict = None):
        """Log an error with full context"""
        self.error_log.append({
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        })

    def update_node_stats(self, node_type: str, execution_time: float, memory_used: float, had_error: bool = False):
        """Update statistics for a node type"""
        stats = self.node_statistics[node_type]
        stats['executions'] += 1
        stats['total_time'] += execution_time
        stats['avg_memory'] = (stats['avg_memory'] * (stats['executions'] - 1) + memory_used) / stats['executions']
        if had_error:
            stats['errors'] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected data"""
        return {
            'execution_count': len(self.execution_logs),
            'performance_samples': sum(len(v) for v in self.performance_data.values()),
            'memory_snapshots': len(self.memory_snapshots),
            'errors': len(self.error_log),
            'node_types_tracked': len(self.node_statistics),
            'total_runtime': time.time() - self.start_time
        }

    def clear(self):
        """Clear all collected data"""
        self.execution_logs.clear()
        self.performance_data.clear()
        self.memory_snapshots.clear()
        self.error_log.clear()
        self.node_statistics.clear()
        self.start_time = time.time()


# Global collector instance
_global_collector = DebugCollector()


class QwenDebugController:
    """
    Comprehensive debug controller for Qwen node system.
    Controls debug levels, collects statistics, and provides analysis.
    """

    CATEGORY = "QwenImage/Debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "debug_level": (["off", "basic", "verbose", "trace"], {
                    "default": "off"
                }),
                "collect_stats": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
                "profile_performance": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
                "track_memory": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
                "show_system_info": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
            },
            "optional": {
                "component_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty for all, or use: encoder,template,vae"
                }),
                "log_filter_pattern": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Regex pattern to filter logs (optional)"
                }),
                "export_format": (["none", "json", "text"], {
                    "default": "none"
                }),
                "clear_history": ("BOOLEAN", {
                    "default": False,
                    "label_on": "clear",
                    "label_off": "keep"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("debug_info", "stats_json", "performance_json")
    FUNCTION = "control_debug"
    OUTPUT_NODE = True

    def control_debug(
        self,
        debug_level: str,
        collect_stats: bool,
        profile_performance: bool,
        track_memory: bool,
        show_system_info: bool,
        component_filter: str = "",
        log_filter_pattern: str = "",
        export_format: str = "none",
        clear_history: bool = False
    ) -> Tuple[str, str, str]:
        """Main debug control function"""

        global _global_collector

        # Clear history if requested
        if clear_history:
            _global_collector.clear()
            QwenLogger.clear_decision_history()

        # Set debug levels based on selection
        self._configure_debug_level(debug_level)

        # Collect memory snapshot if tracking
        if track_memory:
            _global_collector.log_memory_snapshot()

        # Build component filter list
        components = [c.strip() for c in component_filter.split(",")] if component_filter else []

        # Compile log filter regex if provided
        log_pattern = None
        if log_filter_pattern:
            try:
                log_pattern = re.compile(log_filter_pattern)
            except re.error as e:
                QwenLogger.error(f"Invalid regex pattern: {e}")

        # Generate debug output
        debug_info = self._generate_debug_info(
            debug_level=debug_level,
            show_system_info=show_system_info,
            components=components,
            log_pattern=log_pattern
        )

        # Generate statistics
        stats_data = {}
        if collect_stats:
            stats_data = self._generate_statistics()

        # Generate performance data
        perf_data = {}
        if profile_performance:
            perf_data = self._generate_performance_data()

        # Export if requested
        if export_format == "json":
            self._export_json(debug_info, stats_data, perf_data)
        elif export_format == "text":
            self._export_text(debug_info)

        # Convert to JSON strings for output
        stats_json = json.dumps(stats_data, indent=2) if stats_data else "{}"
        perf_json = json.dumps(perf_data, indent=2) if perf_data else "{}"

        return (debug_info, stats_json, perf_json)

    def _configure_debug_level(self, level: str):
        """Configure debug levels across all systems"""

        # Configure debug_patch verbose mode
        if level in ["verbose", "trace"]:
            debug_patch.set_debug_verbose(True)
            QwenLogger.info(f"Debug level set to: {level.upper()}")
        else:
            debug_patch.set_debug_verbose(False)

        # Configure QwenLogger log level
        if level == "trace":
            QwenLogger.set_log_level("DEBUG")
        elif level == "verbose":
            QwenLogger.set_log_level("INFO")
        elif level == "basic":
            QwenLogger.set_log_level("WARNING")
        else:  # off
            QwenLogger.set_log_level("ERROR")

    def _generate_debug_info(
        self,
        debug_level: str,
        show_system_info: bool,
        components: List[str],
        log_pattern: Optional[re.Pattern]
    ) -> str:
        """Generate comprehensive debug information"""

        lines = []
        lines.append("=" * 80)
        lines.append("QWEN DEBUG CONTROLLER - System Analysis")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Debug Level: {debug_level.upper()}")
        lines.append("")

        # System information
        if show_system_info:
            lines.extend(self._format_system_info())
            lines.append("")

        # Collector summary
        summary = _global_collector.get_summary()
        lines.append("COLLECTION SUMMARY:")
        lines.append("-" * 80)
        for key, value in summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Node statistics
        if _global_collector.node_statistics:
            lines.append("NODE STATISTICS:")
            lines.append("-" * 80)
            for node_type, stats in sorted(_global_collector.node_statistics.items()):
                if not components or any(c in node_type.lower() for c in components):
                    avg_time = stats['total_time'] / stats['executions'] if stats['executions'] > 0 else 0
                    lines.append(f"  {node_type}:")
                    lines.append(f"    Executions: {stats['executions']}")
                    lines.append(f"    Total Time: {stats['total_time']:.4f}s")
                    lines.append(f"    Avg Time: {avg_time:.4f}s")
                    lines.append(f"    Avg Memory: {stats['avg_memory']:.2f} MB")
                    lines.append(f"    Errors: {stats['errors']}")
            lines.append("")

        # Memory snapshots
        if _global_collector.memory_snapshots:
            lines.append("MEMORY USAGE:")
            lines.append("-" * 80)
            recent_snapshots = _global_collector.memory_snapshots[-5:]  # Last 5
            for snapshot in recent_snapshots:
                lines.append(f"  [{snapshot['timestamp']}] RSS: {snapshot['rss_mb']:.2f} MB, "
                           f"VMS: {snapshot['vms_mb']:.2f} MB, Usage: {snapshot['percent']:.2f}%")
            lines.append("")

        # Execution logs (filtered)
        if _global_collector.execution_logs and debug_level in ["verbose", "trace"]:
            lines.append("EXECUTION LOGS:")
            lines.append("-" * 80)
            recent_logs = _global_collector.execution_logs[-50:]  # Last 50
            for log in recent_logs:
                # Apply component filter
                if components and not any(c in log['component'].lower() for c in components):
                    continue
                # Apply regex filter
                if log_pattern and not log_pattern.search(log['message']):
                    continue

                lines.append(f"  [{log['timestamp']}] [{log['level']}] {log['component']}")
                lines.append(f"    {log['message']}")
            lines.append("")

        # Smart decision history
        decision_history = QwenLogger.get_decision_history()
        if decision_history and debug_level in ["verbose", "trace"]:
            lines.append("SMART LABELING DECISIONS:")
            lines.append("-" * 80)
            for decision in decision_history[-20:]:  # Last 20
                lines.append(f"  [{decision['timestamp']}] {decision['decision']}")
                lines.append(f"    Reason: {decision['reason']}")
                if decision['details']:
                    lines.append(f"    Details: {decision['details']}")
            lines.append("")

        # Errors
        if _global_collector.error_log:
            lines.append("ERRORS:")
            lines.append("-" * 80)
            for error in _global_collector.error_log[-10:]:  # Last 10
                lines.append(f"  [{error['timestamp']}] {error['component']}: {error['error_type']}")
                lines.append(f"    {error['error_message']}")
                if debug_level == "trace":
                    lines.append(f"    Traceback:")
                    for line in error['traceback'].split('\n')[:10]:  # First 10 lines
                        lines.append(f"      {line}")
            lines.append("")

        # Performance insights
        if _global_collector.performance_data:
            lines.extend(self._format_performance_insights())

        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_system_info(self) -> List[str]:
        """Format system information"""
        lines = []
        lines.append("SYSTEM INFORMATION:")
        lines.append("-" * 80)

        # Python info
        lines.append(f"  Python: {sys.version.split()[0]}")

        # PyTorch info
        lines.append(f"  PyTorch: {torch.__version__}")
        lines.append(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"  CUDA Version: {torch.version.cuda}")
            lines.append(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                lines.append(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")

        # CPU info
        lines.append(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, "
                    f"{psutil.cpu_count(logical=True)} logical")

        # Memory info
        mem = psutil.virtual_memory()
        lines.append(f"  System Memory: {mem.total / 1024**3:.2f} GB total, "
                    f"{mem.available / 1024**3:.2f} GB available ({mem.percent}% used)")

        return lines

    def _format_performance_insights(self) -> List[str]:
        """Generate performance insights"""
        lines = []
        lines.append("PERFORMANCE INSIGHTS:")
        lines.append("-" * 80)

        for component, samples in _global_collector.performance_data.items():
            if not samples:
                continue

            durations = [s['duration'] for s in samples]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)

            lines.append(f"  {component}:")
            lines.append(f"    Samples: {len(samples)}")
            lines.append(f"    Avg Duration: {avg_duration:.4f}s")
            lines.append(f"    Min Duration: {min_duration:.4f}s")
            lines.append(f"    Max Duration: {max_duration:.4f}s")

            # Identify slow executions
            slow_threshold = avg_duration * 2
            slow_count = sum(1 for d in durations if d > slow_threshold)
            if slow_count > 0:
                lines.append(f"    WARNING: {slow_count} executions took >2x average time")

        return lines

    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate statistics dictionary"""
        return {
            'summary': _global_collector.get_summary(),
            'node_statistics': dict(_global_collector.node_statistics),
            'memory_current': _global_collector.memory_snapshots[-1] if _global_collector.memory_snapshots else {},
            'error_count': len(_global_collector.error_log),
            'decision_count': len(QwenLogger.get_decision_history())
        }

    def _generate_performance_data(self) -> Dict[str, Any]:
        """Generate performance data dictionary"""
        perf = {}

        for component, samples in _global_collector.performance_data.items():
            if not samples:
                continue

            durations = [s['duration'] for s in samples]
            perf[component] = {
                'count': len(samples),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations),
                'recent_samples': samples[-10:]  # Last 10
            }

        return perf

    def _export_json(self, debug_info: str, stats: Dict, performance: Dict):
        """Export data as JSON file"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'debug_info': debug_info,
                'statistics': stats,
                'performance': performance,
                'execution_logs': _global_collector.execution_logs[-100:],
                'memory_snapshots': _global_collector.memory_snapshots[-20:],
                'errors': _global_collector.error_log[-20:]
            }

            filename = f"/tmp/qwen_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            QwenLogger.info(f"Debug data exported to: {filename}")
        except Exception as e:
            QwenLogger.error(f"Failed to export JSON: {e}")

    def _export_text(self, debug_info: str):
        """Export debug info as text file"""
        try:
            filename = f"/tmp/qwen_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(debug_info)

            QwenLogger.info(f"Debug info exported to: {filename}")
        except Exception as e:
            QwenLogger.error(f"Failed to export text: {e}")


# Utility functions for nodes to integrate with the collector
def log_node_execution(node_type: str, message: str, level: str = "INFO"):
    """Log a node execution event"""
    _global_collector.log_execution(node_type, message, level)


def log_node_performance(node_type: str, duration: float, details: Dict = None):
    """Log node performance metrics"""
    _global_collector.log_performance(node_type, duration, details)


def log_node_error(node_type: str, error: Exception, context: Dict = None):
    """Log a node error"""
    _global_collector.log_error(node_type, error, context)


def update_node_statistics(node_type: str, execution_time: float, memory_used: float, had_error: bool = False):
    """Update statistics for a node"""
    _global_collector.update_node_stats(node_type, execution_time, memory_used, had_error)


# Context manager for automatic performance tracking
class NodeExecutionContext:
    """Context manager for tracking node execution"""

    def __init__(self, node_type: str, collect_memory: bool = False):
        self.node_type = node_type
        self.collect_memory = collect_memory
        self.start_time = None
        self.start_memory = None
        self.had_error = False

    def __enter__(self):
        self.start_time = time.time()
        if self.collect_memory:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        memory_used = 0.0
        if self.collect_memory and self.start_memory is not None:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_used = max(0, current_memory - self.start_memory)

        if exc_type is not None:
            self.had_error = True
            log_node_error(self.node_type, exc_val)

        log_node_performance(self.node_type, duration)
        update_node_statistics(self.node_type, duration, memory_used, self.had_error)

        return False  # Don't suppress exceptions


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenDebugController": QwenDebugController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenDebugController": "Debug Controller"
}