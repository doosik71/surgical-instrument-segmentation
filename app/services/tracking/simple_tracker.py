"""Simple multi-tool tracker for contour-derived tool tips."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.domain.models import Point, ToolGeometry


@dataclass(slots=True)
class TrackState:
    """Mutable state for a tracked tool."""

    track_id: int
    last_tip: Point
    last_center: Point
    missed_frames: int = 0
    trajectory: list[Point] = field(default_factory=list)


class SimpleToolTracker:
    """Nearest-neighbor tracker using tip and center distance."""

    def __init__(self, max_tip_distance: float = 120.0, max_missed_frames: int = 5) -> None:
        self.max_tip_distance = max_tip_distance
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, TrackState] = {}
        self._next_track_id = 1

    def reset(self) -> None:
        """Clear all tracking state."""
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, tools: list[ToolGeometry]) -> list[ToolGeometry]:
        """Assign stable track IDs to the current frame's tools."""
        if not tools:
            self._age_unmatched_tracks(set())
            return tools

        matches, unmatched_tool_indices, unmatched_track_ids = self._match_tools(tools)

        matched_track_ids = set()
        for tool_index, track_id in matches.items():
            tool = tools[tool_index]
            tool.track_id = track_id
            state = self._tracks[track_id]
            state.last_tip = tool.tip
            state.last_center = tool.center
            state.missed_frames = 0
            state.trajectory.append(tool.tip)
            if len(state.trajectory) > 64:
                state.trajectory.pop(0)
            matched_track_ids.add(track_id)

        for tool_index in unmatched_tool_indices:
            tool = tools[tool_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            tool.track_id = track_id
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                last_tip=tool.tip,
                last_center=tool.center,
                missed_frames=0,
                trajectory=[tool.tip],
            )
            matched_track_ids.add(track_id)

        self._age_unmatched_tracks(matched_track_ids)
        return tools

    def get_trajectories(self) -> dict[int, list[Point]]:
        """Return tracked tip trajectories by track ID."""
        return {track_id: state.trajectory[:] for track_id, state in self._tracks.items()}

    def _age_unmatched_tracks(self, matched_track_ids: set[int]) -> None:
        stale_track_ids: list[int] = []
        for track_id, state in self._tracks.items():
            if track_id in matched_track_ids:
                continue
            state.missed_frames += 1
            if state.missed_frames > self.max_missed_frames:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del self._tracks[track_id]

    def _match_tools(self, tools: list[ToolGeometry]) -> tuple[dict[int, int], list[int], list[int]]:
        if not self._tracks:
            return {}, list(range(len(tools))), []

        candidate_pairs: list[tuple[float, int, int]] = []
        for tool_index, tool in enumerate(tools):
            tip_array = np.array(tool.tip, dtype=np.float32)
            center_array = np.array(tool.center, dtype=np.float32)

            for track_id, state in self._tracks.items():
                track_tip_array = np.array(state.last_tip, dtype=np.float32)
                track_center_array = np.array(state.last_center, dtype=np.float32)
                tip_distance = float(np.linalg.norm(tip_array - track_tip_array))
                center_distance = float(np.linalg.norm(center_array - track_center_array))
                score = tip_distance + 0.35 * center_distance
                if score <= self.max_tip_distance:
                    candidate_pairs.append((score, tool_index, track_id))

        candidate_pairs.sort(key=lambda item: item[0])
        matches: dict[int, int] = {}
        matched_tools: set[int] = set()
        matched_tracks: set[int] = set()

        for _, tool_index, track_id in candidate_pairs:
            if tool_index in matched_tools or track_id in matched_tracks:
                continue
            matches[tool_index] = track_id
            matched_tools.add(tool_index)
            matched_tracks.add(track_id)

        unmatched_tool_indices = [index for index in range(len(tools)) if index not in matched_tools]
        unmatched_track_ids = [track_id for track_id in self._tracks if track_id not in matched_tracks]
        return matches, unmatched_tool_indices, unmatched_track_ids
