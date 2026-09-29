"""Local E007 research system; no medical data leaves the process."""

TARGETS = (
    "spine_positioning", "spine_axis", "spine_foreign_object",
    "right_hip_positioning_rotation", "right_hip_roi",
    "left_hip_positioning_rotation", "left_hip_roi",
    "quality_spine", "quality_right_hip", "quality_left_hip",
)
ABLATIONS = {
    "A0": ("dino",), "A1": ("mi2",), "A2": ("geometry",),
    "A3": ("dino", "mi2"), "A4": ("dino", "geometry"),
    "A5": ("mi2", "geometry"), "A6": ("dino", "mi2", "geometry"),
}
