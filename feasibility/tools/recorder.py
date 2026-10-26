"""Record the camera stream to disk (MPEG-TS, stream copy, no re-encoding) for offline processing.

Why MPEG-TS: the camera does not announce its video parameters up front, and a Matroska/MP4 file
written by stream copy ends up with no readable video track. TS works. The camera sends a full
video frame only every ~4 s and nothing decodes before the first one, so a recording is only
"ready" a few seconds after it starts; the web page shows that.

Files go to the preset's output/recordings/: <label>_<YYYYmmdd_HHMMSS>.ts, plus a .json with the
wall-clock start time and how it ended, and a .log with ffmpeg's warnings. The camera URL (which
holds the password) is never written to the JSON or returned by the API.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time

READY_AFTER_S = 6.0
READY_MIN_BYTES = 300_000


class RecorderError(Exception):
    pass


def _clean(label):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", label or "").strip("-")[:40] or "rec"


def _scrub(text):
    return re.sub(r"://[^@/\s]+@", "://***@", text)


class Recorder:
    def __init__(self, url, out_dir, max_minutes=30, min_free_gb=2.0):
        self.url, self.dir = url, out_dir
        self.max_s, self.min_free = max_minutes * 60, min_free_gb
        self.lock = threading.Lock()
        self.proc = None
        self.meta = None
        self.t0 = 0.0
        self.error = None
        os.makedirs(out_dir, exist_ok=True)
        threading.Thread(target=self._watch, daemon=True).start()

    # -- control
    def start(self, label):
        with self.lock:
            if self.proc is not None:
                raise RecorderError("already recording")
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RecorderError("ffmpeg not found (brew install ffmpeg)")
            free = shutil.disk_usage(self.dir).free / 1e9
            if free < self.min_free:
                raise RecorderError(f"only {free:.1f} GB free on the disk, need {self.min_free:.0f} GB")
            label = _clean(label)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            base = os.path.join(self.dir, f"{label}_{stamp}")
            cmd = [ffmpeg, "-y", "-v", "warning"]
            if self.url.startswith("rtsp"):
                cmd += ["-rtsp_transport", "tcp"]
            cmd += ["-i", self.url, "-map", "0", "-c", "copy", "-f", "mpegts", base + ".ts"]
            self.logf = open(base + ".log", "w")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.logf)
            self.t0 = time.time()
            self.error = None
            self.meta = {"label": label, "file": os.path.basename(base) + ".ts", "started_wall": self.t0,
                         "started_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0)),
                         "command": "ffmpeg " + " ".join("<camera url>" if a == self.url else a for a in cmd[1:])}
            self._write_meta()
            return self._status_locked()

    def stop(self, reason="stopped"):
        with self.lock:
            return self._stop_locked(reason)

    def _stop_locked(self, reason):
        if self.proc is None:
            return self._status_locked()
        p = self.proc
        try:
            p.stdin.write(b"q")           # ffmpeg finishes the file cleanly on "q"
            p.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=4)
            except subprocess.TimeoutExpired:
                p.kill()
        self._finalize(reason, p.returncode)
        return self._status_locked()

    def _finalize(self, reason, code):
        path = os.path.join(self.dir, self.meta["file"])
        end = time.time()
        self.meta.update(ended_wall=end, duration_s=round(end - self.t0, 2), exit_code=code, reason=reason,
                         bytes=os.path.getsize(path) if os.path.exists(path) else 0)
        self._write_meta()
        try:
            self.logf.close()
        except Exception:
            pass
        self.proc = None

    def _write_meta(self):
        with open(os.path.join(self.dir, self.meta["file"][:-3] + ".json"), "w") as f:
            json.dump(self.meta, f, indent=1)

    def _watch(self):
        """Stop at the time limit, and notice if ffmpeg died on its own."""
        while True:
            time.sleep(1.5)
            with self.lock:
                if self.proc is None:
                    continue
                code = self.proc.poll()
                if code is not None:
                    self.error = self._log_tail() or f"ffmpeg exited with code {code}"
                    self._finalize(f"ffmpeg exited ({code})", code)
                elif time.time() - self.t0 > self.max_s:
                    self._stop_locked(f"time limit ({self.max_s // 60:.0f} min)")

    def _log_tail(self):
        try:
            with open(os.path.join(self.dir, self.meta["file"][:-3] + ".log")) as f:
                lines = [l.strip() for l in f if l.strip() and "PPS" not in l and "no frame" not in l
                         and "decode_slice" not in l and "Last message" not in l]
            return _scrub(" | ".join(lines[-3:]))[:300]
        except OSError:
            return ""

    # -- state
    def status(self):
        with self.lock:
            return self._status_locked()

    def _status_locked(self):
        free = shutil.disk_usage(self.dir).free / 1e9
        st = {"recording": self.proc is not None, "free_gb": round(free, 1), "error": self.error,
              "max_minutes": self.max_s // 60, "recent": self.recent()}
        if self.proc is not None:
            path = os.path.join(self.dir, self.meta["file"])
            size = os.path.getsize(path) if os.path.exists(path) else 0
            el = time.time() - self.t0
            st.update(label=self.meta["label"], file=self.meta["file"], elapsed_s=round(el, 1), bytes=size,
                      ready=el >= READY_AFTER_S and size >= READY_MIN_BYTES)
        return st

    def recent(self, n=8):
        out = []
        for name in sorted(os.listdir(self.dir), reverse=True)[:200]:
            if name.endswith(".json"):
                try:
                    m = json.load(open(os.path.join(self.dir, name)))
                except (OSError, ValueError):
                    continue
                if "ended_wall" not in m and not (self.proc and self.meta and m.get("file") == self.meta["file"]):
                    m["reason"] = "unfinished (the server stopped mid-recording)"
                out.append({k: m.get(k) for k in ("label", "file", "started_iso", "duration_s", "bytes", "reason")})
        out.sort(key=lambda m: m["started_iso"] or "", reverse=True)
        return out[:n]
