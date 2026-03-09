---
name: workflow-json
description: Help write and review ComfyUI Looper workflow JSON files using best practices and BKM conventions. Use when designing new workflows, reviewing JSON, writing expressions, or choosing parameter values.
---

# Workflow JSON Helper

When helping with ComfyUI Looper workflow JSON files, always reference [BKM.md](BKM.md) for detailed conventions, patterns, parameter ranges, and rules.

## Quick Reference

- **Schema**: Each workflow has `all_settings` (array of LoopSettings) and `version` (always 1)
- **Inheritance**: Omitted fields inherit backward from the previous section. Use this to reduce redundancy.
- **Expressions**: Numeric fields can be strings with math expressions using variables `n`, `offset`, `total_n`
- **Transforms**: Applied in sequence; order matters. Combine carefully.

When writing or reviewing a workflow JSON, validate against the rules and patterns in the BKM.
