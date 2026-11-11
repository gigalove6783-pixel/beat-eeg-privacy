"""Independent, dataset-agnostic attacks on transmitted outputs; never imports BEAT."""

from .api import AuditConfig, Condition, Population, Segment, audit_stream, audit_timing

__all__ = ["AuditConfig", "Condition", "Population", "Segment", "audit_stream", "audit_timing"]
__version__ = "0.1.0"
