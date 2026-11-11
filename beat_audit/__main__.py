"""``python -m beat_audit`` has the same interface as the installed console command."""

from .cli import main

raise SystemExit(main())
