"""Recoup harness — rolls a case forward over the horizon under a given policy,
resolving recovery and revocation as competing risks, and scores batches.

The harness is the ONLY component that touches `simulator/response.py`. Policies
receive the observable `case` and their own state, never the hidden truth.
"""
