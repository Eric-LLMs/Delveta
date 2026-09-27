"""ToolIntentModel backends — model access ONLY.

A backend turns (query, facts, candidate cards) into the raw reply dict
{capability_id, confidence, arguments} or raises
:class:`..base.ToolIntentUnavailable`. No backend controls the funnel: no
routing, no Binder call, no fallback decision, no second model hop — the
ladder lives in :mod:`..model`, the correctness gate in :mod:`..parser`.
"""
