from __future__ import annotations

from daphne_core.v2.pfd_alignment import (
    extract_pfd_directive_addressees,
    infer_responder_aliases,
)


def test_extract_addressees_agency_subject_pattern():
    text = "The Secretary of State for the Home Department should direct that the systems are monitored."
    addrs = extract_pfd_directive_addressees(text)
    assert any("Secretary of State" in a for a in addrs)


def test_extract_addressees_prepositional_target_pattern():
    text = "I request that this issue is addressed by OHFT in its response to this Report."
    addrs = extract_pfd_directive_addressees(text)
    assert "OHFT" in addrs


def test_extract_addressees_compound_coordination_pattern():
    text = "I encourage the Secretary of State for Justice and the Chief Constable of Thames Valley Police to work together."
    addrs = extract_pfd_directive_addressees(text)
    assert any("Secretary of State for Justice" in a for a in addrs)
    assert any("Chief Constable" in a for a in addrs)


def test_infer_responder_aliases_picks_up_org_and_acronyms():
    text = """
Thames Valley Police

Yours sincerely,

Chief Constable
TVP
"""
    aliases = infer_responder_aliases(text)
    assert "Thames Valley Police" in aliases
    assert "Chief Constable" in aliases
    assert "TVP" in aliases

