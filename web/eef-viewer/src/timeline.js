export function nearestFrame(times, time) {
  if (!times.length) return -1;
  let low = 0, high = times.length - 1;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (times[mid] < time) low = mid + 1; else high = mid;
  }
  return low && time - times[low - 1] <= times[low] - time ? low - 1 : low;
}

export function nextAnomaly(indices, current) {
  return indices.find(index => index > current) ?? indices[0] ?? current;
}

export function waitEvent(element, event, timeout = 15000, signal) {
  return new Promise((resolve, reject) => {
    const finish = (error) => {
      clearTimeout(timer); element.removeEventListener(event, success); element.removeEventListener('error', failure);
      signal?.removeEventListener('abort', abort);
      error ? reject(error) : resolve();
    };
    const success = () => finish();
    const failure = () => finish(new Error('视频无法解码或读取失败，可切换 H.264 预览'));
    const abort = () => finish(new DOMException('Video load cancelled', 'AbortError'));
    const timer = setTimeout(() => finish(new Error('视频定位超时；画面尚未同步')), timeout);
    element.addEventListener(event, success, { once: true });
    element.addEventListener('error', failure, { once: true });
    if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, { once: true });
  });
}

export async function seekVideo(video, time, signal) {
  if (signal?.aborted) throw new DOMException('Video load cancelled', 'AbortError');
  if (video.readyState < 1) await waitEvent(video, 'loadedmetadata', 15000, signal);
  // Float32 dataset timestamps can sit just before the encoded frame PTS.
  // Seek slightly inside that frame, rather than accidentally showing its predecessor.
  const target = Math.min(Math.max(0, time + 0.0001), Math.max(0, video.duration - 0.00001));
  if (Math.abs(video.currentTime - target) < 0.00001 && video.readyState >= 2) return;
  const ready = waitEvent(video, 'seeked', 15000, signal);
  video.currentTime = target;
  await ready;
}

export class VideoClock {
  constructor(video, onFrame) {
    this.video = video; this.onFrame = onFrame; this.presentedTime = null; this.frameCount = 0;
    this.callback = (_, metadata) => {
      this.presentedTime = metadata.mediaTime; this.frameCount = metadata.presentedFrames;
      onFrame(metadata.mediaTime, true);
      this.id = video.requestVideoFrameCallback(this.callback);
    };
    this.fallback = () => onFrame(video.currentTime, false);
    if (video.requestVideoFrameCallback) this.id = video.requestVideoFrameCallback(this.callback);
    else video.addEventListener('timeupdate', this.fallback);
  }
  dispose() {
    if (this.id !== undefined) this.video.cancelVideoFrameCallback(this.id);
    this.video.removeEventListener('timeupdate', this.fallback);
  }
}
