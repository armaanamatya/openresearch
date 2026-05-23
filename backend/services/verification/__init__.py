"""Verification service namespace.

The active verification records live in ``backend.schemas.verifications`` and
``backend.persistence.repositories.verification_repository``.  This package is
kept as the stable service import surface promised by the app skeleton tests.
"""

from backend.persistence.repositories.verification_repository import VerificationRepository
from backend.schemas.verifications import VerificationRecord, VerificationStatus

__all__ = ["VerificationRecord", "VerificationRepository", "VerificationStatus"]
