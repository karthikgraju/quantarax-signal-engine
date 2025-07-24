const CACHE_NAME = 'qx-cache-v1';
const ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './logo-192.png',
  './logo-512.png'
];

// Cache core assets on install
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

// Remove old caches on activate
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.map(k => {
        if (k !== CACHE_NAME) return caches.delete(k);
      }))
    ).then(() => self.clients.claim())
  );
});

// Serve from cache, fallback to network
self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(cached =>
      cached || fetch(e.request)
    )
  );
});
