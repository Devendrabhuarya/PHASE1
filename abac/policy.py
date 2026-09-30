"""
Minimal ABAC policy engine (Phase 0 stub).

This is a plain-Python stand-in for the "Pre-Training ABAC Gate" and
"Update Authorization" steps in the architecture diagram. It will be
expanded in later phases (reputation-aware checks get added once the
anomaly-detection / reputation modules exist).
"""

from dataclasses import dataclass, field


@dataclass
class ClientAttributes:
    client_id: str
    role: str                  # owner / family_member / guest / service_provider
    device_type: str           # lock / camera / thermostat / sensor
    location: str = "unknown"
    trust_score: float = 0.8   # 0.0-1.0, default starting reputation
    status: str = "active"     # active / revoked


# very small default policy: role must be in an allowed set, status must
# be active. Extend this as your project grows (time windows, location
# checks, resource sensitivity, etc.)
ALLOWED_ROLES = {"owner", "family_member", "guest", "service_provider"}


def pre_training_gate(attrs: ClientAttributes) -> bool:
    """
    Step 1: Pre-training ABAC gate.
    Returns True (ALLOW) or False (DENY).
    """
    if attrs.status != "active":
        return False
    if attrs.role not in ALLOWED_ROLES:
        return False
    if attrs.trust_score <= 0.0:
        return False
    return True


def update_authorization_gate(attrs: ClientAttributes, session_valid: bool = True) -> bool:
    """
    Step 3: Update authorization & validation.
    Returns True (ALLOW) or False (DENY).
    """
    if not session_valid:
        return False
    if attrs.status != "active":
        return False
    return True


if __name__ == "__main__":
    a = ClientAttributes(client_id="c1", role="owner", device_type="lock")
    print("pre_training_gate:", pre_training_gate(a))
    print("update_authorization_gate:", update_authorization_gate(a))

    b = ClientAttributes(client_id="c2", role="owner", device_type="camera", status="revoked")
    print("revoked client pre_training_gate:", pre_training_gate(b))
