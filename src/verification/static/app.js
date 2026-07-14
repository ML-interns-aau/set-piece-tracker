const videoOriginal = document.getElementById("video-original");
const videoOverlay = document.getElementById("video-overlay");
const clipSelect = document.getElementById("clip-select");
const eventSelect = document.getElementById("event-select");
const reviewerInput = document.getElementById("reviewer");
const metadataPanel = document.getElementById("metadata-panel");
const speedSelect = document.getElementById("speed-select");

let events = [];
let currentIndex = 0;
let fps = 25.0;
let syncing = false;

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return res.json();
}

function estimateFps(evts) {
  const withFrame = evts.find((e) => e.frame > 0 && e.timestamp_s > 0);
  return withFrame ? withFrame.frame / withFrame.timestamp_s : 25.0;
}

async function loadClips() {
  const clips = await fetchJSON("/api/clips");
  clipSelect.innerHTML = "";
  for (const clipId of clips) {
    const option = document.createElement("option");
    option.value = clipId;
    option.textContent = clipId;
    clipSelect.appendChild(option);
  }
  if (clips.length) await loadClip(clips[0]);
}

async function loadClip(clipId) {
  videoOriginal.src = `/video/original/${encodeURIComponent(clipId)}`;
  videoOverlay.src = `/video/overlay/${encodeURIComponent(clipId)}`;
  await loadEvents(clipId);
}

async function loadEvents(clipId) {
  events = await fetchJSON(`/api/events?clip_id=${encodeURIComponent(clipId)}`);
  fps = estimateFps(events);

  eventSelect.innerHTML = "";
  events.forEach((event, i) => {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = `frame ${event.frame} - ${event.event_type}` +
      (event.reviewed ? " (reviewed)" : "");
    eventSelect.appendChild(option);
  });

  currentIndex = events.findIndex((e) => !e.reviewed);
  if (currentIndex === -1) currentIndex = 0;
  jumpToCurrentEvent();
}

function currentEvent() {
  return events[currentIndex] || null;
}

function jumpToCurrentEvent() {
  const event = currentEvent();
  if (!event) return;
  eventSelect.value = String(currentIndex);
  const t = event.timestamp_s;
  videoOriginal.currentTime = t;
  videoOverlay.currentTime = t;
  renderMetadata(event);
}

function renderMetadata(event) {
  metadataPanel.innerHTML = "";
  const rows = [
    ["Clip ID", event.clip_id],
    ["Frame Number", event.frame],
    ["Event Type", event.event_type],
    ["Confidence", event.confidence?.toFixed?.(2) ?? event.confidence],
    ["Reliability Score", event.reliability_score ?? "n/a"],
    ["Predicted Team", event.team ?? "n/a"],
    ["Player IDs", (event.player_ids || []).join(", ") || "none"],
    ["Timestamp", `${event.timestamp_s.toFixed(2)}s`],
    ["Reviewed", event.reviewed ? "yes" : "no"],
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    metadataPanel.append(dt, dd);
  }
}

function stepFrame(delta) {
  const dt = delta / fps;
  videoOriginal.currentTime = Math.max(0, videoOriginal.currentTime + dt);
  videoOverlay.currentTime = videoOriginal.currentTime;
}

function togglePlay() {
  if (videoOriginal.paused) {
    videoOriginal.play();
    videoOverlay.play();
  } else {
    videoOriginal.pause();
    videoOverlay.pause();
  }
}

function moveToEvent(delta) {
  if (!events.length) return;
  currentIndex = Math.min(Math.max(currentIndex + delta, 0), events.length - 1);
  jumpToCurrentEvent();
}

function nextUnreviewedOrAdvance() {
  const next = events.findIndex((e, i) => i > currentIndex && !e.reviewed);
  currentIndex = next !== -1 ? next : Math.min(currentIndex + 1, events.length - 1);
  jumpToCurrentEvent();
}

async function submitVerdict(verdict) {
  const event = currentEvent();
  if (!event) return;
  await fetchJSON("/api/verdict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      clip_id: event.clip_id,
      frame: event.frame,
      event: event.event_type,
      verdict,
      reviewer: reviewerInput.value || "manual",
    }),
  });
  await loadEvents(event.clip_id);
  nextUnreviewedOrAdvance();
}

function linkVideos(leader, follower) {
  leader.addEventListener("timeupdate", () => {
    if (syncing) return;
    if (Math.abs(leader.currentTime - follower.currentTime) > 0.05) {
      syncing = true;
      follower.currentTime = leader.currentTime;
      syncing = false;
    }
  });
}
linkVideos(videoOriginal, videoOverlay);
linkVideos(videoOverlay, videoOriginal);

document.getElementById("btn-play").addEventListener("click", togglePlay);
document.getElementById("btn-prev-frame").addEventListener("click", () => stepFrame(-1));
document.getElementById("btn-next-frame").addEventListener("click", () => stepFrame(1));
document.getElementById("btn-prev-event").addEventListener("click", () => moveToEvent(-1));
document.getElementById("btn-next-event").addEventListener("click", () => moveToEvent(1));
document.getElementById("btn-verified").addEventListener("click", () => submitVerdict("confirm"));
document.getElementById("btn-rejected").addEventListener("click", () => submitVerdict("reject"));
document.getElementById("btn-edit").addEventListener("click", () => submitVerdict("edit"));

clipSelect.addEventListener("change", () => loadClip(clipSelect.value));
eventSelect.addEventListener("change", () => {
  currentIndex = Number(eventSelect.value);
  jumpToCurrentEvent();
});
speedSelect.addEventListener("change", () => {
  const rate = Number(speedSelect.value);
  videoOriginal.playbackRate = rate;
  videoOverlay.playbackRate = rate;
});

document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT") return;
  switch (ev.key) {
    case " ":
      ev.preventDefault();
      togglePlay();
      break;
    case "ArrowLeft":
      ev.preventDefault();
      stepFrame(-1);
      break;
    case "ArrowRight":
      ev.preventDefault();
      stepFrame(1);
      break;
    case "v":
    case "V":
      submitVerdict("confirm");
      break;
    case "r":
    case "R":
      submitVerdict("reject");
      break;
    case "e":
    case "E":
      submitVerdict("edit");
      break;
  }
});

loadClips().catch((err) => console.error("Failed to load clips:", err));
