import supervision as sv
import numpy as np
class FootballTracker:
    BALL_TRACKER_ID = -99
    def __init__(
        self,
        track_thresh: float = 0.20,
        track_buffer: int = 90,
        match_thresh: float = 0.80,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            minimum_consecutive_frames=1,
        )
    def update(self, detections: sv.Detections) -> sv.Detections:
        player_mask = detections.class_id == 0
        ball_mask = detections.class_id == 32
        player_detections = detections[player_mask]
        ball_detections = detections[ball_mask]
        tracked_players = self.tracker.update_with_detections(player_detections)
        if len(ball_detections) > 0:
            if len(ball_detections) > 1:
                best_idx = int(np.argmax(ball_detections.confidence))
                ball_detections = ball_detections[[best_idx]]
            ball_detections.tracker_id = np.array([self.BALL_TRACKER_ID])
            tracked_players.data = {}
            ball_detections.data = {}
            return sv.Detections.merge([tracked_players, ball_detections])
        return tracked_players
