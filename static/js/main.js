// Air Quality ML Dashboard — main.js

document.addEventListener('DOMContentLoaded', () => {

  // ── Active nav link highlight ─────────────────────
  const links = document.querySelectorAll('.nav-link');
  const current = window.location.pathname;
  links.forEach(link => {
    if (link.getAttribute('href') === current) {
      link.classList.add('active');
    }
  });

  // ── Mobile sidebar toggle ─────────────────────────
  const toggleBtn = document.getElementById('sidebar-toggle');
  const sidebar   = document.getElementById('sidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }

  // ── Image modal ───────────────────────────────────
  const imgCards = document.querySelectorAll('.img-card img');
  imgCards.forEach(img => {
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', () => openModal(img.src, img.alt));
  });

  function openModal(src, caption) {
    const overlay = document.createElement('div');
    overlay.id = 'img-modal';
    overlay.style.cssText = `
      position:fixed;inset:0;background:rgba(0,0,0,.85);
      display:flex;align-items:center;justify-content:center;
      z-index:9999;cursor:zoom-out;flex-direction:column;gap:12px;
    `;
    const img = document.createElement('img');
    img.src = src;
    img.style.cssText = 'max-width:90vw;max-height:80vh;border-radius:8px;box-shadow:0 8px 40px rgba(0,0,0,.5)';
    const cap = document.createElement('p');
    cap.textContent = caption || '';
    cap.style.cssText = 'color:#e2e8f0;font-size:.9rem;';
    overlay.appendChild(img);
    overlay.appendChild(cap);
    overlay.addEventListener('click', () => overlay.remove());
    document.body.appendChild(overlay);
  }

  // ── Dataset search / filter ───────────────────────
  const searchInput = document.getElementById('table-search');
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      const q = searchInput.value.toLowerCase();
      document.querySelectorAll('#data-table tbody tr').forEach(row => {
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
      });
    });
  }

  // ── Prediction form AQI category display ─────────
  const predForm = document.getElementById('pred-form');
  if (predForm) {
    predForm.addEventListener('submit', e => {
      e.preventDefault();
      const pm25 = parseFloat(document.getElementById('pm25')?.value || 0);
      const category = getAQICategory(pm25);
      const resultBox = document.getElementById('pred-result');
      if (resultBox) {
        document.getElementById('pred-value').textContent = pm25.toFixed(1);
        document.getElementById('pred-category').textContent = category.label;
        document.getElementById('pred-category').style.color = category.color;
        resultBox.style.display = 'block';
      }
    });
  }

  function getAQICategory(pm25) {
    if (pm25 <= 12)  return { label: 'Good',          color: '#22c55e' };
    if (pm25 <= 35)  return { label: 'Moderate',       color: '#f59e0b' };
    if (pm25 <= 55)  return { label: 'Unhealthy (Sensitive)', color: '#f97316' };
    if (pm25 <= 150) return { label: 'Unhealthy',      color: '#ef4444' };
    if (pm25 <= 250) return { label: 'Very Unhealthy', color: '#a855f7' };
    return { label: 'Hazardous', color: '#7f1d1d' };
  }

  // ── Smooth scroll for anchor links ───────────────
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

});
