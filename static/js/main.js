/* AERO-PURE main.js v2.0 */

document.addEventListener('DOMContentLoaded', () => {

  /* ── Sidebar toggle ──────────────────────────────── */
  const menuBtn = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  if (menuBtn && sidebar) {
    menuBtn.addEventListener('click', () => sidebar.classList.toggle('open'));
    document.addEventListener('click', e => {
      if (sidebar.classList.contains('open') &&
          !sidebar.contains(e.target) && e.target !== menuBtn)
        sidebar.classList.remove('open');
    });
  }

  /* ── Active nav link ─────────────────────────────── */
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(a => {
    const href = a.getAttribute('href');
    if (href === path || (href !== '/' && path.startsWith(href)))
      a.classList.add('active');
  });

  /* ── Live clock ──────────────────────────────────── */
  const clock = document.getElementById('live-clock');
  if (clock) {
    const tick = () => {
      clock.textContent = new Date().toLocaleTimeString('en-GB',
        { hour: '2-digit', minute: '2-digit' });
    };
    tick();
    setInterval(tick, 15000);
  }

  /* ── Image modal with keyboard + prev/next ───────── */
  let modalImages = [];
  let modalIndex  = 0;

  const backdrop = document.getElementById('img-modal');
  const modalImg = document.getElementById('modal-img');
  const modalCap = document.getElementById('modal-caption');

  function openModal(imgs, idx) {
    modalImages = imgs;
    modalIndex  = idx;
    showModalImage();
    backdrop && backdrop.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    backdrop && backdrop.classList.remove('open');
    document.body.style.overflow = '';
  }

  function showModalImage() {
    if (!modalImg || !modalImages.length) return;
    const item = modalImages[modalIndex];
    modalImg.src = item.src;
    modalImg.alt = item.alt;
    if (modalCap) modalCap.textContent = item.alt;
  }

  // Collect all chart images on page
  const allCards = Array.from(document.querySelectorAll('.img-card'));
  const imgData  = allCards.map(card => {
    const img = card.querySelector('img');
    return { src: img ? img.src : '', alt: img ? img.alt : '' };
  });

  allCards.forEach((card, i) => {
    card.addEventListener('click', () => openModal(imgData, i));
  });

  document.getElementById('modal-close')
    ?.addEventListener('click', closeModal);
  backdrop?.addEventListener('click', e => {
    if (e.target === backdrop) closeModal();
  });
  document.getElementById('modal-prev')
    ?.addEventListener('click', () => {
      modalIndex = (modalIndex - 1 + modalImages.length) % modalImages.length;
      showModalImage();
    });
  document.getElementById('modal-next')
    ?.addEventListener('click', () => {
      modalIndex = (modalIndex + 1) % modalImages.length;
      showModalImage();
    });
  document.addEventListener('keydown', e => {
    if (!backdrop?.classList.contains('open')) return;
    if (e.key === 'Escape')     closeModal();
    if (e.key === 'ArrowLeft') {
      modalIndex = (modalIndex - 1 + modalImages.length) % modalImages.length;
      showModalImage();
    }
    if (e.key === 'ArrowRight') {
      modalIndex = (modalIndex + 1) % modalImages.length;
      showModalImage();
    }
  });

  /* ── Table search ────────────────────────────────── */
  const tableSearch = document.getElementById('table-search');
  if (tableSearch) {
    tableSearch.addEventListener('input', () => {
      const q = tableSearch.value.toLowerCase().trim();
      document.querySelectorAll('#data-table tbody tr').forEach(row => {
        row.style.display =
          (!q || row.textContent.toLowerCase().includes(q)) ? '' : 'none';
      });
    });
  }

  /* ── Prediction AQI client display ──────────────── */
  const predForm = document.getElementById('pred-form');
  if (predForm) {
    // form submits normally to server — no JS override needed
    // just show loading state
    predForm.addEventListener('submit', () => {
      const btn = predForm.querySelector('button[type="submit"]');
      if (btn) { btn.textContent = '⏳ Predicting…'; btn.disabled = true; }
    });
  }

  /* ── Stat counter animation ──────────────────────── */
  document.querySelectorAll('[data-count]').forEach(el => {
    const target  = parseFloat(el.dataset.count);
    const suffix  = el.dataset.suffix || '';
    const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
    if (isNaN(target)) return;
    let start = null;
    const duration = 800;
    const step = ts => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const val = target * ease;
      el.textContent = val.toFixed(decimals)
        .replace(/\B(?=(\d{3})+(?!\d))/g, ',') + suffix;
      if (progress < 1) requestAnimationFrame(step);
      else el.textContent = target.toLocaleString() + suffix;
    };
    requestAnimationFrame(step);
  });

  /* ── Copy-to-clipboard for code blocks ──────────── */
  document.querySelectorAll('code[data-copy]').forEach(el => {
    el.style.cursor = 'pointer';
    el.title = 'Click to copy';
    el.addEventListener('click', () => {
      navigator.clipboard?.writeText(el.textContent).then(() => {
        const orig = el.style.background;
        el.style.background = '#d1fae5';
        setTimeout(() => el.style.background = orig, 1000);
      });
    });
  });

});
