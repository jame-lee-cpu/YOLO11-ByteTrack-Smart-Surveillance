const API = {
  // 页面脚本统一走这里发 JSON 请求，避免每个模板重复写 fetch 配置。
  async post(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: body ? JSON.stringify(body) : undefined
    });
    const data = await res.json().catch(() => ({ok: false, message: `HTTP ${res.status}`}));
    if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
    return data;
  },

  async get(url) {
    const res = await fetch(url);
    const data = await res.json().catch(() => ({ok: false, message: `HTTP ${res.status}`}));
    if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
    return data;
  }
};

const UI = {
  showMessage(el, text, isError = false) {
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('err-text', isError);
  },

  toggleMaximize(card, btn) {
    if (!card || !btn) return;
    card.classList.toggle('is-maximized');
    btn.textContent = card.classList.contains('is-maximized') ? '还原' : '最大化';
  }
};

const TrailControl = {
  parseIds(value) {
    return String(value || '').split(',').map(v => v.trim()).filter(Boolean).map(Number).filter(Number.isInteger);
  },

  async apply(taskId, msgEl) {
    try {
      const modeEl = document.getElementById('runtime-trail-mode');
      const idsEl = document.getElementById('runtime-highlight-ids');
      const lenEl = document.getElementById('runtime-trail-len');
      if (!modeEl || !idsEl || !lenEl) throw new Error('缺少必要元素');

      const data = await API.post(`/tasks/${taskId}/trail-control`, {
        trail_mode: modeEl.value,
        highlight_ids: idsEl.value,
        trail_len: lenEl.value
      });
      UI.showMessage(msgEl, data.message || '轨迹设置已应用', !data.ok);
    } catch (err) {
      UI.showMessage(msgEl, err.message || '请求失败', true);
    }
  },

  async pause(taskId, msgEl, statusEl) {
    try {
      const data = await API.post(`/tasks/${taskId}/pause`);
      UI.showMessage(msgEl, data.message || '任务已暂停', !data.ok);
      if (data.ok && statusEl) statusEl.textContent = '任务已暂停，点击继续分析后恢复处理。';
    } catch (err) {
      UI.showMessage(msgEl, err.message || '请求失败', true);
    }
  },

  async resume(taskId, msgEl, statusEl) {
    try {
      const data = await API.post(`/tasks/${taskId}/resume`);
      UI.showMessage(msgEl, data.message || '任务已继续', !data.ok);
      if (data.ok && statusEl) statusEl.textContent = '任务处理中，页面每2秒自动检查状态。';
    } catch (err) {
      UI.showMessage(msgEl, err.message || '请求失败', true);
    }
  }
};

const PreviewSelection = {
  imagePoint(event, img) {
    // MJPEG 预览图使用 object-fit: contain，显示区域可能有留边。
    // 这里先扣掉留边，再把浏览器坐标换算成原始视频帧坐标。
    const rect = img.getBoundingClientRect();
    const naturalW = img.naturalWidth || rect.width;
    const naturalH = img.naturalHeight || rect.height;
    const naturalRatio = naturalW / Math.max(1, naturalH);
    const boxRatio = rect.width / Math.max(1, rect.height);
    let drawW = rect.width;
    let drawH = rect.height;
    let offsetX = 0;
    let offsetY = 0;

    if (boxRatio > naturalRatio) {
      drawW = rect.height * naturalRatio;
      offsetX = (rect.width - drawW) / 2;
    } else {
      drawH = rect.width / naturalRatio;
      offsetY = (rect.height - drawH) / 2;
    }

    const px = event.clientX - rect.left - offsetX;
    const py = event.clientY - rect.top - offsetY;
    if (px < 0 || py < 0 || px > drawW || py > drawH) return null;

    return {
      x: Math.round(px * naturalW / Math.max(1, drawW)),
      y: Math.round(py * naturalH / Math.max(1, drawH))
    };
  },

  appendHighlightId(trackId) {
    // 后端已经更新运行时高亮集合，这里同步右侧输入框，便于用户看见当前选择。
    const input = document.getElementById('runtime-highlight-ids');
    if (!input) return;
    const ids = input.value.split(',').map((item) => item.trim()).filter(Boolean);
    if (!ids.includes(String(trackId))) ids.push(String(trackId));
    input.value = ids.join(',');

    const mode = document.getElementById('runtime-trail-mode');
    if (mode && mode.value === 'none') mode.value = 'selected';
  },

  removeHighlightId(trackId) {
    const input = document.getElementById('runtime-highlight-ids');
    if (!input) return;
    const ids = input.value.split(',').map((item) => item.trim()).filter(Boolean);
    input.value = ids.filter((id) => id !== String(trackId)).join(',');
  },

  async select(taskId, point, action = 'select') {
    const res = await fetch(`/tasks/${taskId}/click-select`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...point, action})
    });
    const data = await res.json().catch(() => ({ok: false, message: `HTTP ${res.status}`}));
    if (!res.ok && data.ok !== false) data.ok = false;
    return data;
  },

  bind(taskId, img, msgEl) {
    if (!img) return;
    let clickTimer = null;
    const handlePoint = async (event, action) => {
      const point = this.imagePoint(event, img);
      if (!point) return;
      try {
        const data = await this.select(taskId, point, action);
        UI.showMessage(msgEl, data.message || '点击选择已提交', !data.ok);
        if (data.ok && data.track_id !== undefined) {
          if (action === 'unselect') this.removeHighlightId(data.track_id);
          else this.appendHighlightId(data.track_id);
        }
      } catch (err) {
        UI.showMessage(msgEl, err.message || '点击选择失败', true);
      }
    };

    img.addEventListener('click', (event) => {
      // 浏览器双击会先触发单击；短延迟可以避免“双击取消”前先执行“单击选中”。
      if (clickTimer) window.clearTimeout(clickTimer);
      clickTimer = window.setTimeout(() => {
        clickTimer = null;
        handlePoint(event, 'select');
      }, 220);
    });
    img.addEventListener('dblclick', (event) => {
      event.preventDefault();
      if (clickTimer) {
        window.clearTimeout(clickTimer);
        clickTimer = null;
      }
      handlePoint(event, 'unselect');
    });
  }
};

const LineEditor = {
  xyxyToEditable(line) {
    const x1 = Number(line.x1), y1 = Number(line.y1), x2 = Number(line.x2), y2 = Number(line.y2);
    const dx = x2 - x1, dy = y2 - y1;
    return {
      center_x: (x1 + x2) / 2,
      center_y: (y1 + y2) / 2,
      length: Math.max(20, Math.hypot(dx, dy)),
      angle: Math.atan2(dy, dx) * 180 / Math.PI
    };
  },

  editableToXyxy(line) {
    const rad = line.angle * Math.PI / 180;
    const dx = Math.cos(rad) * line.length / 2;
    const dy = Math.sin(rad) * line.length / 2;
    return {
      x1: Math.round(line.center_x - dx),
      y1: Math.round(line.center_y - dy),
      x2: Math.round(line.center_x + dx),
      y2: Math.round(line.center_y + dy)
    };
  },

  scale(line, fromW, fromH, toW, toH) {
    return this.xyxyToEditable({
      x1: line.x1 * toW / fromW,
      y1: line.y1 * toH / fromH,
      x2: line.x2 * toW / fromW,
      y2: line.y2 * toH / fromH
    });
  },

  clamp(line, maxW, maxH) {
    line.center_x = Math.max(0, Math.min(maxW, line.center_x));
    line.center_y = Math.max(0, Math.min(maxH, line.center_y));
    line.length = Math.max(20, Math.min(Math.max(maxW, maxH) * 1.5, line.length));
    if (line.angle > 180) line.angle -= 360;
    if (line.angle < -180) line.angle += 360;
  }
};
