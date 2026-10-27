import { validateBundle, validateSnapshot } from './data.js';

export function artifactUrl(ref, base, origin) {
  if (!ref || !/^[a-f0-9]{64}$/.test(ref.sha256)
      || !new RegExp(`^objects/${ref.sha256.slice(0, 2)}/${ref.sha256}\\.[a-z0-9]+$`).test(ref.path)) {
    throw new Error('无效的内容引用');
  }
  const url = new URL(ref.path.split('/').at(-1), new URL(base, origin));
  if (url.origin !== new URL(origin).origin) throw new Error('数据 URL 必须与页面同源');
  return url.href;
}

export async function expandManifest(manifest, url, fetcher = fetch) {
  if (manifest.schema_version !== 5) return validateBundle(manifest);
  if (!manifest.artifact_base || !Array.isArray(manifest.episodes)) throw new Error('缺少已注册的共享数据路径');
  const get = async (ref) => {
    const response = await fetcher(artifactUrl(ref, manifest.artifact_base, url));
    if (!response.ok) throw new Error(`数据加载失败：HTTP ${response.status}`);
    return response.json();
  };
  const [reference, shards] = await Promise.all([
    get(manifest.reference_snapshot), Promise.all(manifest.episodes.map(async (episode) => {
      const shard = await get(episode.events);
      if (shard.episode_id !== episode.episode_id || !Array.isArray(shard.events)
          || shard.events.some((event) => event.episode_id !== episode.episode_id)) throw new Error('Episode 数据不一致');
      return shard.events;
    })),
  ]);
  return validateBundle({ ...manifest, reference_snapshot: reference, events: shards.flat() });
}

export function snapshotLoader(fetcher = fetch) {
  const cache = new Map();
  return async (event, bundle, url) => {
    if (event.snapshot) return validateSnapshot(event.snapshot, bundle.metadata);
    if (!event.snapshot_ref) return null;
    const path = artifactUrl(event.snapshot_ref, bundle.artifact_base, url);
    if (!cache.has(path)) {
      const request = fetcher(path).then((response) => {
        if (!response.ok) throw new Error(`姿态加载失败：HTTP ${response.status}`);
        return response.json();
      });
      cache.set(path, request);
      request.catch(() => cache.delete(path));
    }
    return validateSnapshot(await cache.get(path), bundle.metadata);
  };
}
