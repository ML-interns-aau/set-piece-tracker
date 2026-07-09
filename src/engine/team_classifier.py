from __future__ import annotations
import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from collections import defaultdict, deque
from math import ceil
class TeamClassifier:
    REFEREE_ID = -2
    UNKNOWN_ID = -1
    GK_ID = -3
    _GREEN_LOWER = np.array([35, 40, 40])
    _GREEN_UPPER = np.array([85, 255, 255])
    def __init__(
        self,
        n_teams: int = 2,
        history_len: int = 15,
        refit_interval: int = 150,
        detect_goalkeeper: bool = True,
        gk_min_cluster_samples: int = 4,
        gk_min_separation_ratio: float = 2.0,
    ):
        self.n_teams = n_teams
        self.history_len = history_len
        self.refit_interval = refit_interval
        self.kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
        self.is_fitted = False
        self.outlier_threshold = 100.0
        self.vote_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=history_len))
        self.locked_teams: dict[int, int] = {}
        self.switch_votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(6, history_len // 2)))
        self.team_lock_min_votes = max(6, history_len // 2)
        self.frame_count = 0
        # --- Goalkeeper extension (FR-005) ------------------------------------
        # detect_goalkeeper=False reproduces the original 2-team-only behavior
        # exactly (gk_cluster_idx always None, no fallback pass ever runs).
        self.detect_goalkeeper = detect_goalkeeper
        self.gk_min_cluster_samples = gk_min_cluster_samples
        self.gk_min_separation_ratio = gk_min_separation_ratio
        self.gk_cluster_idx = None  # index into self.kmeans.cluster_centers_ identified as GK, when the kit-color fit separates cleanly; None otherwise
        self._cluster_to_team: dict[int, int] = {i: i for i in range(n_teams)}
        self.gk_confidence: dict[int, str] = {}  # tracker_id -> "high" (kit-clustered) / "low" (position fallback)
    def _get_jersey_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        crop_h = crop.shape[0]
        return crop[:max(1, crop_h // 2), :]
    def _extract_dominant_hsv(self, bgr_crop: np.ndarray) -> np.ndarray:
        if bgr_crop.size == 0:
            return np.zeros(3)
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        mask_h = (pixels[:, 0] >= self._GREEN_LOWER[0]) & (pixels[:, 0] <= self._GREEN_UPPER[0])
        mask_s = (pixels[:, 1] >= self._GREEN_LOWER[1]) & (pixels[:, 1] <= self._GREEN_UPPER[1])
        mask_v = (pixels[:, 2] >= self._GREEN_LOWER[2]) & (pixels[:, 2] <= self._GREEN_UPPER[2])
        green_mask = mask_h & mask_s & mask_v
        non_green = pixels[~green_mask]
        if len(non_green) < 5:
            non_green = pixels
        bright_mask = non_green[:, 2] > 30
        bright = non_green[bright_mask]
        if len(bright) < 5:
            bright = non_green
        km = KMeans(n_clusters=1, n_init=3, random_state=42)
        km.fit(bright)
        return km.cluster_centers_[0]
    @staticmethod
    def _foot_point(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, y2])
    def fit_teams(self, frame: np.ndarray, detections: sv.Detections):
        player_mask = detections.class_id == 0
        player_detections = detections[player_mask]
        if len(player_detections) < 6:
            return
        colors = []
        for bbox in player_detections.xyxy:
            crop = self._get_jersey_crop(frame, bbox)
            if crop.size == 0:
                continue
            domcol = self._extract_dominant_hsv(crop)
            colors.append(domcol)
        if len(colors) < self.n_teams:
            return
        colors = np.asarray(colors)
        kmeans2 = KMeans(n_clusters=self.n_teams, n_init=10, random_state=42)
        labels2 = kmeans2.fit_predict(colors)
        chosen_kmeans, chosen_labels = kmeans2, labels2
        self.gk_cluster_idx = None
        self._cluster_to_team = {i: i for i in range(self.n_teams)}
        
        if self.detect_goalkeeper and len(colors) >= max(self.n_teams + 1, self.gk_min_cluster_samples):
            n_gk_clusters = self.n_teams + 1
            dists2 = np.linalg.norm(colors - kmeans2.cluster_centers_[labels2], axis=1)
            typical_spread = max(float(np.percentile(dists2, 85)), 1e-6)
            kmeans3 = KMeans(n_clusters=n_gk_clusters, n_init=10, random_state=42)
            labels3 = kmeans3.fit_predict(colors)
            counts = np.bincount(labels3, minlength=n_gk_clusters)
            if len(set(labels3)) == n_gk_clusters:
                gk_idx = int(np.argmin(counts))
                is_minority = counts[gk_idx] <= max(1, len(colors) * 0.4)
                other_idxs = [idx for idx in range(n_gk_clusters) if idx != gk_idx]
                separation = min(
                    float(np.linalg.norm(kmeans3.cluster_centers_[gk_idx] - kmeans3.cluster_centers_[j]))
                    for j in other_idxs
                )
                separation_ratio = separation / typical_spread
                if is_minority and separation_ratio >= self.gk_min_separation_ratio:
                    outfield_idxs = sorted(other_idxs)
                    chosen_kmeans, chosen_labels = kmeans3, labels3
                    self.gk_cluster_idx = gk_idx
                    self._cluster_to_team = {idx: team for team, idx in enumerate(outfield_idxs)}
        self.kmeans = chosen_kmeans
        centers = self.kmeans.cluster_centers_
        dists = np.linalg.norm(colors - centers[chosen_labels], axis=1)
        self.outlier_threshold = np.percentile(dists, 85) * 2.0
        self.is_fitted = True
    def _apply_position_fallback(self, raw_teams: dict, foot_points: dict) -> int | None:
        
        by_team: dict = defaultdict(list)
        for i, team in raw_teams.items():
            if team == self.REFEREE_ID or i not in foot_points:
                continue
            by_team[team].append(i)
        best_i, best_score = None, -1.0
        for indices in by_team.values():
            if len(indices) < 3:
                continue
            pts = np.array([foot_points[i] for i in indices])
            for local_idx, i in enumerate(indices):
                others = np.delete(pts, local_idx, axis=0)
                isolation = float(np.mean(np.linalg.norm(others - pts[local_idx], axis=1)))
                if isolation > best_score:
                    best_score, best_i = isolation, i
        return best_i
    def assign_teams(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        team_ids = np.full(len(detections), self.UNKNOWN_ID, dtype=int)
        if not self.is_fitted or (self.frame_count > 0 and self.frame_count % self.refit_interval == 0):
            self.fit_teams(frame, detections)
        self.frame_count += 1
        if not self.is_fitted:
            return team_ids
        fallback_active = self.detect_goalkeeper and self.gk_cluster_idx is None
        raw_teams: dict = {}
        tracker_ids: dict = {}
        foot_points: dict = {}
        for i, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            if class_id != 0:
                continue
            tracker_id = (int(detections.tracker_id[i]) if detections.tracker_id is not None else None)
            crop = self._get_jersey_crop(frame, bbox)
            if crop.size == 0:
                continue
            domcol = self._extract_dominant_hsv(crop)
            dists = np.linalg.norm(self.kmeans.cluster_centers_ - domcol, axis=1)
            min_dist = np.min(dists)
            if min_dist > self.outlier_threshold:
                raw_team = self.REFEREE_ID
            else:
                cluster_idx = int(np.argmin(dists))
                if self.gk_cluster_idx is not None and cluster_idx == self.gk_cluster_idx:
                    raw_team = self.GK_ID
                else:
                    raw_team = self._cluster_to_team.get(cluster_idx, cluster_idx)
            raw_teams[i] = raw_team
            tracker_ids[i] = tracker_id
            if fallback_active:
                foot_points[i] = self._foot_point(bbox)
        if fallback_active:
            gk_i = self._apply_position_fallback(raw_teams, foot_points)
            if gk_i is not None:
                raw_teams[gk_i] = self.GK_ID
        for i, raw_team in raw_teams.items():
            tracker_id = tracker_ids[i]
            if tracker_id is not None:
                self.vote_history[tracker_id].append(raw_team)
                history = list(self.vote_history[tracker_id])
                voted_team = max(set(history), key=history.count)
                if voted_team == self.REFEREE_ID:
                    team_ids[i] = self.REFEREE_ID
                else:
                    if tracker_id not in self.locked_teams:
                        if len(history) >= self.team_lock_min_votes and history.count(voted_team) >= ceil(0.7 * len(history)):
                            self.locked_teams[tracker_id] = voted_team
                    if tracker_id in self.locked_teams:
                        current_team = self.locked_teams[tracker_id]
                        if voted_team != current_team:
                            self.switch_votes[tracker_id].append(voted_team)
                            votes = list(self.switch_votes[tracker_id])
                            if len(votes) >= 6 and votes.count(voted_team) >= 5:
                                self.locked_teams[tracker_id] = voted_team
                                self.switch_votes[tracker_id].clear()
                                team_ids[i] = voted_team
                            else:
                                team_ids[i] = current_team
                        else:
                            self.switch_votes[tracker_id].clear()
                            team_ids[i] = current_team
                    else:
                        team_ids[i] = voted_team
                if team_ids[i] == self.GK_ID:
                    self.gk_confidence[tracker_id] = "high" if self.gk_cluster_idx is not None else "low"
                else:
                    self.gk_confidence.pop(tracker_id, None)
            else:
                team_ids[i] = raw_team
        return team_ids
