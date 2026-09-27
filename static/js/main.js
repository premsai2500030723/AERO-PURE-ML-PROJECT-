// Air Quality ML Dashboard — main.js  (UI v2)

document.addEventListener('DOMContentLoaded', () => {

  // ── Live clock ────────────────────────────────────
  const clock = document.getElementById('clock');
  if (clock) {
    const tick = () => {
      const now = new Date();
      clock.textContent = now.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit' });
    };
    tick();
    setInterval(tick, 10000);
  }

  // ── Mobile sidebar toggle ─────────────────────────
  const toggleBtn = document.getElementById('sidebar-toggle');
  const sidebar   = document.getElementById('sidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => sidebar.classList.toggle('open'));
    // close on overlay click
    document.addEventListener('click', e => {
      if (sidebar.classList.contains('open') &&
          !sidebar.contains(e.target) &&
          e.target !== toggleBtn) {
        sidebar.classList.remove('open');
      }
    });
  }

  // ── Active nav link highlight ─────────────────────
  const current = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('href') === current) link.classList.add('active');
  });

  // ── Image zoom modal ──────────────────────────────
  document.querySelectorAll('.img-card').forEach(card => {
    card.addEventListener('click', () => {
      const img = card.querySelector('img');
      const label = card.querySelector('.img-card-body span')?.textContent || '';
      openModal(img.src, label);
    });
  });

  function openModal(src, caption) {
    // Remove existing modal if any
    document.getElementById('img-modal')?.remove();

    const overlay = document.createElement('div');
    overlay.id = 'img-modal';
    overlay.style.cssText = `
      position:fixed;inset:0;background:rgba(10,10,20,.88);
      display:flex;align-items:center;justify-content:center;
      z-index:9999;cursor:zoom-out;flex-direction:column;gap:14px;
      backdrop-filter:blur(6px);animation:fadeIn .2s ease;
    `;

    const imgEl = document.createElement('img');
    imgEl.src = src;
    imgEl.style.cssText = `
      max-width:92vw;max-height:78vh;border-radius:10px;
      box-shadow:0 20px 60px rgba(0,0,0,.6);
    `;

    const cap = document.createElement('p');
    cap.textContent = caption;
    cap.style.cssText = 'color:#cbd5e1;font-size:.88rem;font-weight:600;';

    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕ Close';
    closeBtn.style.cssText = `
      background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
      color:#fff;padding:7px 18px;border-radius:8px;cursor:pointer;font-size:.8rem;
    `;
    closeBtn.onclick = () => overlay.remove();

    overlay.append(imgEl, cap, closeBtn);
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
  }

  // Inject fadeIn keyframe once
  if (!document.getElementById('modal-style')) {
    const s = document.createElement('style');
    s.id = 'modal-style';
    s.textContent = '@keyframes fadeIn{from{opacity:0}to{opacity:1}}';
    document.head.appendChild(s);
  }

  // ── Dataset table live search ─────────────────────
  const searchInput = document.getElementById('table-search');
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      const q = searchInput.value.toLowerCase().trim();
      document.querySelectorAll('#data-table tbody tr').forEach(row => {
        row.style.display = (!q || row.textContent.toLowerCase().includes(q)) ? '' : 'none';
      });
    });
  }

  // ── Prediction form — client AQI lookup ──────────
  const predForm = document.getElementById('pred-form');
  if (predForm) {
    predForm.addEventListener('submit', e => {
      const pm25Input = document.getElementById('pm25');
      if (!pm25Input || !pm25Input.value) return; // let server handle

      const val = parseFloat(pm25Input.value);
      if (isNaN(val)) return;

      const resultBox = document.getElementById('pred-result');
      if (!resultBox) return;

      e.preventDefault();
      const cat = getAQICategory(val);
      document.getElementById('pred-value').textContent    = val.toFixed(1);
      const catEl = document.getElementById('pred-category');
      catEl.textContent  = cat.label;
      catEl.style.color  = cat.color;
      resultBox.style.display = 'block';
      resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  }

  function getAQICategory(pm25) {
    if (pm25 <= 12)  return { label: '🟢 Good',                        color: '#22c55e' };
    if (pm25 <= 35)  return { label: '🟡 Moderate',                    color: '#eab308' };
    if (pm25 <= 55)  return { label: '🟠 Unhealthy for Sensitive',     color: '#f97316' };
    if (pm25 <= 150) return { label: '🔴 Unhealthy',                   color: '#ef4444' };
    if (pm25 <= 250) return { label: '🟣 Very Unhealthy',              color: '#a855f7' };
    return           { label: '⚫ Hazardous',                          color: '#450a0a' };
  }

  // ── Smooth anchor scrolling ───────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth' }); }
    });
  });

  // ── Stat counter animation ────────────────────────
  document.querySelectorAll('.stat-value').forEach(el => {
    const raw = el.textContent.trim();
    const num = parseFloat(raw.replace(/[^0-9.]/g, ''));
    if (isNaN(num) || num === 0) return;
    const suffix = raw.replace(/[0-9.,]/g, '').trim();
    const decimals = (raw.includes('.')) ? (raw.split('.')[1]?.replace(/\D/g,'').length || 0) : 0;
    let start = 0;
    const duration = 900;
    const step = timestamp => {
      if (!start) start = timestamp;
      const progress = Math.min((timestamp - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = (num * ease).toFixed(decimals)
        .replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (suffix ? ' ' + suffix : '');
      if (progress < 1) requestAnimationFrame(step);
      else el.textContent = raw; // restore original
    };
    requestAnimationFrame(step);
  });

});
