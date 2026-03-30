"""Data Intelligence skill handler — dataset search, analysis, and processing."""

from __future__ import annotations

from typing import Any


class DataIntelHandler:
    """Handles data intelligence operations in 3 modes."""

    def __init__(self, cache_dir: str = ""):
        self.cache_dir = cache_dir

    def handle_fixed_mode(self, data_path: str, task: str = "") -> dict:
        """Mode 1: Validate, profile, and prepare existing data.

        Returns structured result with validation, profile, and recommendations.
        """
        from autotrainer.managers.data_manager import DataManager

        dm = DataManager(self.cache_dir)

        # Step 1: Validate
        validation = dm.validate_dataset(data_path)

        # Step 2: Profile
        profile = dm.profile_dataset(data_path)

        # Step 3: Recommendations
        recommendations = []
        if not validation["valid"]:
            recommendations.append("Fix data format errors before training")
        if profile.has_images and not profile.quality_flags:
            pass  # OK
        if profile.num_samples < 100:
            recommendations.append("Dataset is very small — consider adding more data")
        if profile.text_lengths.get("avg", 0) > 2000:
            recommendations.append("Average text length is long — consider increasing max_seq_len")

        return {
            "action": "validate_and_profile",
            "validation": validation,
            "profile": profile.to_dict(),
            "recommendations": recommendations,
            "data_profile": {
                "num_samples": profile.num_samples,
                "avg_text_len": profile.text_lengths.get("avg", 0),
                "has_images": profile.has_images,
                "quality_score": 1.0 if validation["valid"] else 0.5,
            },
        }

    def handle_expand_mode(self, existing_paths: list[str], task: str = "") -> dict:
        """Mode 2: Find additional datasets to complement existing ones."""
        from autotrainer.managers.data_manager import DataManager

        dm = DataManager(self.cache_dir)

        # Search HF
        hf_results = dm.search_hf(query=f"{task} training dataset", limit=10)

        return {
            "action": "expand",
            "existing_datasets": [dm.profile_dataset(p).to_dict() for p in existing_paths],
            "candidates": hf_results,
            "recommendations": [
                "Review candidates and select datasets with complementary data",
                "Check format compatibility before downloading",
            ],
        }

    def handle_discover_mode(self, task: str, requirements: str = "") -> dict:
        """Mode 3: Discover datasets from scratch."""
        from autotrainer.managers.data_manager import DataManager

        dm = DataManager(self.cache_dir)

        # Multi-source search
        query = f"{task} {requirements}".strip()
        hf_results = dm.search_hf(query=query, limit=10)

        return {
            "action": "discover",
            "query": query,
            "candidates": hf_results,
            "recommendations": [
                "Select datasets that match your task",
                "Download and validate before training",
            ],
        }
