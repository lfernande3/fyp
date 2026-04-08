"""
Receiver Models for Advanced Collision Resolution (O8)

Provides three receiver variants that replace the binary collision-detection
rule in the Simulator:

- COLLISION  : baseline — success if and only if exactly one node transmits.
- CAPTURE    : probabilistic capture — the strongest signal is decoded if its
               SINR exceeds a threshold γ, even when multiple nodes transmit.
- SIC        : successive interference cancellation — signals are decoded
               iteratively from strongest to weakest; each decoded signal is
               subtracted from the interference pool before the next attempt.

Usage
-----
from src.receiver_models import ReceiverModel, resolve_transmissions

successful = resolve_transmissions(
    transmitters,                        # list of Node objects that transmitted
    model=ReceiverModel.SIC,
    capture_threshold=10.0,              # SINR threshold for CAPTURE (linear ratio)
    sic_sinr_threshold=1.0,              # SINR threshold for SIC
)
# successful is the (possibly empty) list of Node objects whose packets
# were successfully decoded this slot.
"""

import numpy as np
from enum import Enum
from typing import List, Any


class ReceiverModel(Enum):
    """Receiver model variants."""
    COLLISION = "collision"   # Baseline: success iff exactly 1 transmitter
    CAPTURE   = "capture"     # Probabilistic capture by strongest signal
    SIC       = "sic"         # Successive interference cancellation


def resolve_transmissions(
    transmitters: List[Any],
    model: ReceiverModel = ReceiverModel.COLLISION,
    capture_threshold: float = 10.0,
    sic_sinr_threshold: float = 1.0,
) -> List[Any]:
    """
    Determine which nodes succeed in a given slot.

    Parameters
    ----------
    transmitters : list
        Node objects that attempted transmission this slot.
    model : ReceiverModel
        Receiver variant to apply.
    capture_threshold : float
        For CAPTURE model: minimum SINR (linear power ratio) required for
        the dominant signal to be decoded.  Default 10.0 (≈10 dB).
    sic_sinr_threshold : float
        For SIC model: minimum SINR required to decode each successive
        signal.  Default 1.0 (0 dB).

    Returns
    -------
    list
        Subset of ``transmitters`` whose packets are successfully decoded.
        Empty list on complete collision / decoding failure.
    """
    if not transmitters:
        return []

    if model == ReceiverModel.COLLISION:
        # Classic slotted-Aloha: exactly one transmitter means success
        return list(transmitters) if len(transmitters) == 1 else []

    # Draw independent random received powers ~ Exp(1) for each transmitter
    powers = np.random.exponential(1.0, size=len(transmitters))

    if model == ReceiverModel.CAPTURE:
        return _resolve_capture(transmitters, powers, capture_threshold)

    if model == ReceiverModel.SIC:
        return _resolve_sic(transmitters, powers, sic_sinr_threshold)

    # Fallback to collision model
    return list(transmitters) if len(transmitters) == 1 else []


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_capture(transmitters, powers, threshold):
    """Capture effect: strongest signal wins if SINR > threshold."""
    best_idx = int(np.argmax(powers))
    total_power = float(np.sum(powers))
    best_power = float(powers[best_idx])
    interference = total_power - best_power

    if interference < 1e-15:
        # Trivially only one effective power — always succeeds
        return [transmitters[best_idx]]

    sinr = best_power / interference
    if sinr > threshold:
        return [transmitters[best_idx]]
    return []


def _resolve_sic(transmitters, powers, sinr_threshold):
    """
    SIC: decode signals greedily from strongest to weakest.

    For signal i (sorted by power descending):
      SINR_i = P_i / (sum of remaining powers excluding P_i)
    If SINR_i >= sinr_threshold: decode, subtract P_i from interference pool.
    Otherwise: stop (weaker signals have even lower SINR).
    """
    sorted_idx = list(np.argsort(powers)[::-1])   # descending by power
    decoded = []
    remaining_interference = float(np.sum(powers))

    for idx in sorted_idx:
        p_i = float(powers[idx])
        interference_for_i = remaining_interference - p_i

        sinr = p_i / max(interference_for_i, 1e-15)
        if sinr >= sinr_threshold:
            decoded.append(transmitters[idx])
            remaining_interference -= p_i   # subtract decoded signal
        else:
            # If the strongest remaining signal can't be decoded, no weaker
            # one can either — exit early.
            break

    return decoded
