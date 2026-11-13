import { seekVideo, waitEvent, VideoClock } from './timeline.js';

export function selectCameraSlots(cameras, preferredHead) {
  const side = name => cameras.find(camera => camera.camera === `observation.images.${name}`)
    || cameras.find(camera => new RegExp(`(?:^|\\.)(?:hand_|wrist_)${name}$`).test(camera.camera));
  const left = side('left'), right = side('right');
  const external = cameras.filter(camera => camera !== left && camera !== right);
  const heads = external.length ? external : cameras;
  const head = heads.find(camera => camera.camera === preferredHead)
    || heads.find(camera => /(?:^|\.)head(?:_(?:left|right))?$/.test(camera.camera))
    || heads.find(camera => camera.camera.endsWith('.base_view_front')) || heads[0] || null;
  return { head, left: left !== head ? left || null : null,
    right: right !== head ? right || null : null, heads };
}

function delay(ms, signal) {
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(new DOMException('Video load cancelled', 'AbortError')); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, ms);
    if (signal.aborted) abort(); else signal.addEventListener('abort', abort, { once: true });
  });
}

export class VideoGroup {
  constructor(elements, { onFrame, onStatus }) {
    this.onFrame = onFrame; this.onStatus = onStatus; this.rate = 1; this.epoch = 0;
    this.slots = Object.fromEntries(Object.entries(elements).map(([name, video]) => [name, { name, video, state: 'empty', presented: null }]));
    for (const slot of Object.values(this.slots)) {
      slot.video.addEventListener('waiting', () => {
        if (slot.state === 'ready' && this.playIntent) {
          slot.buffering = true;
          if (slot.name === 'head') for (const follower of this.followers) follower.video.pause();
          this.onStatus();
        }
      });
      slot.video.addEventListener('playing', () => {
        slot.buffering = false;
        if (slot.name === 'head' && this.playIntent) for (const follower of this.followers)
          follower.video.play().catch(error => this.playError(follower, error));
        this.onStatus();
      });
    }
    this.head.video.addEventListener('ended', () => this.pause());
  }
  get head() { return this.slots.head; }
  get followers() { return [this.slots.left, this.slots.right].filter(slot => slot.state === 'ready'); }
  get ready() { return Object.values(this.slots).filter(slot => slot.state === 'ready'); }
  get allReady() { return Object.values(this.slots).every(slot => ['ready', 'missing', 'error'].includes(slot.state)); }
  clear() {
    this.epoch++; this.scope?.abort(); this.pause();
    for (const slot of Object.values(this.slots)) {
      slot.clock?.dispose(); slot.clock = null; slot.state = 'empty'; slot.camera = null; slot.presented = null;
      slot.buffering = false; slot.error = null; slot.video.removeAttribute('src'); slot.video.load();
    }
    this.onStatus();
  }
  pause() { this.playIntent = false; for (const slot of Object.values(this.slots)) slot.video.pause(); }
  setRate(rate) { this.rate = rate; for (const slot of Object.values(this.slots)) slot.video.playbackRate = rate; }
  async preview(camera, signal) {
    for (;;) {
      const response = await fetch(camera.preview_url, { method: 'HEAD', signal });
      if (response.status === 200) return camera.preview_url;
      if (response.status !== 202) throw new Error('H.264 预览生成失败');
      await delay(1000, signal);
    }
  }
  async load(cameras, preferredHead, time = 0, forcePreview = false) {
    this.clear(); const epoch = this.epoch;
    this.scope = new AbortController(); const signal = this.scope.signal;
    const selected = selectCameraSlots(cameras, preferredHead);
    const tasks = Object.values(this.slots).map(async slot => {
      slot.camera = selected[slot.name]; slot.state = slot.camera ? 'loading' : 'missing'; this.onStatus();
      if (!slot.camera) return;
      try {
        slot.derived = forcePreview || slot.camera.requires_preview || slot.camera.codec !== 'h264';
        if (slot.derived) slot.state = 'preview'; this.onStatus();
        let url = slot.derived ? await this.preview(slot.camera, signal) : slot.camera.url;
        const open = async () => {
          if (epoch !== this.epoch) return;
          slot.state = 'loading'; this.onStatus();
          const loaded = waitEvent(slot.video, 'loadeddata', 30000, signal);
          slot.video.src = url; slot.video.playbackRate = this.rate; slot.video.load();
          await loaded;
        };
        try { await open(); } catch (error) {
          if (slot.derived || signal.aborted) throw error;
          slot.derived = true; slot.state = 'preview'; this.onStatus();
          url = await this.preview(slot.camera, signal); await open();
        }
        if (epoch !== this.epoch) return;
        slot.state = 'ready';
        slot.clock = new VideoClock(slot.video, (time, supported) => {
          if (epoch !== this.epoch || slot.state !== 'ready') return;
          slot.presented = time; slot.supported = supported;
          if (slot.name === 'head') {
            this.onFrame(time, supported);
            if (this.playIntent && !this.seeking) for (const follower of this.followers) {
              const limit = Math.max(.08, 2 / follower.camera.fps);
              if (!follower.video.seeking && Math.abs(follower.video.currentTime - time) > limit)
                follower.video.currentTime = Math.min(time + .0001, follower.video.duration - .00001);
            }
          }
          this.onStatus();
        });
        await seekVideo(slot.video, time, signal);
        if (epoch === this.epoch) this.onStatus();
      } catch (error) {
        if (epoch !== this.epoch || signal.aborted) return;
        slot.state = 'error'; slot.error = error.message; this.onStatus();
      }
    });
    await Promise.all(tasks);
    return epoch === this.epoch;
  }
  async seek(time) {
    this.pause(); this.seeking = true;
    const epoch = this.epoch;
    try { await Promise.all(this.ready.map(slot => seekVideo(slot.video, time, this.scope?.signal))); }
    finally { if (epoch === this.epoch) this.seeking = false; }
  }
  playError(slot, error) {
    if (!this.playIntent || error.name === 'AbortError') return;
    slot.error = error.message;
    if (slot.name === 'head') this.pause();
    this.onStatus();
  }
  async play() {
    if (!this.allReady) throw new Error('视频仍在准备中');
    if (this.head.state !== 'ready') throw new Error('主相机视频尚未就绪');
    await this.seek(this.head.video.currentTime);
    this.playIntent = true;
    await Promise.all(this.ready.map(slot => slot.video.play().catch(error => this.playError(slot, error))));
  }
  renewLeases() {
    for (const slot of this.ready) if (slot.derived)
      fetch(slot.camera.preview_url.replace('/preview/', '/lease/')).catch(() => {});
  }
}
