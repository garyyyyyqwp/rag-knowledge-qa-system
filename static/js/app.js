/**
 * Knowledge QA — Personal Knowledge Base Q&A System
 * Frontend SPA: Document Management + RAG Chat (SSE Streaming)
 * Vanilla JS — no frameworks, no dependencies
 */

/* ========================================================================
 * 1. DOM HELPERS
 * ======================================================================== */

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

/* ========================================================================
 * 2. STATE MANAGER (with sessionStorage persistence)
 * ======================================================================== */

const STATE_KEY = 'kqa_state';

/** Load persisted state from sessionStorage, falling back to defaults. */
function loadPersistedState() {
  try {
    const saved = sessionStorage.getItem(STATE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      return {
        tab: parsed.tab || 'documents',
        mode: parsed.mode || 'rag',
        documents: [],
        isUploading: false,
        isStreaming: false,
        abortController: null,
        // Per-mode chat history — each mode has its own message list
        chatHistory: parsed.chatHistory || { rag: [], bare: [], compare: [] },
      };
    }
  } catch {
    // sessionStorage unavailable or corrupted — use defaults
  }
  return {
    tab: 'documents',
    mode: 'rag',
    documents: [],
    isUploading: false,
    isStreaming: false,
    abortController: null,
    chatHistory: { rag: [], bare: [], compare: [] },
  };
}

/** Persist essential state fields to sessionStorage. */
function persistState() {
  try {
    // Only persist lightweight fields; skip heavy DOM references
    sessionStorage.setItem(STATE_KEY, JSON.stringify({
      tab: state.tab,
      mode: state.mode,
      chatHistory: state.chatHistory,
    }));
  } catch {
    // Silently ignore — non-critical
  }
}

const state = loadPersistedState();

/* ========================================================================
 * 3. HASH-BASED ROUTING
 * ======================================================================== */

/**
 * Sync URL hash with current tab.
 * When the user switches tabs, update the hash.
 * When the hash changes (browser back/forward), switch to the matching tab.
 */
function syncHashFromTab() {
  const expected = `#${state.tab}`;
  if (window.location.hash !== expected) {
    // Use replaceState to avoid creating a new history entry
    history.replaceState(null, '', expected);
  }
}

function syncTabFromHash() {
  const hash = window.location.hash.replace('#', '');
  if (hash === 'qa') {
    switchTab('qa', false); // don't push hash again
  } else {
    switchTab('documents', false);
  }
}

window.addEventListener('hashchange', syncTabFromHash);

/* ========================================================================
 * 4. TOAST SYSTEM
 * ======================================================================== */

/**
 * Show a toast notification.
 * @param {string} message — the message text
 * @param {'success'|'error'|'warning'} type — toast variant
 */
function toast(message, type = 'success') {
  const container = $('#toastContainer');
  if (!container) return;

  // Enforce max 3 toasts — remove oldest if exceeding
  const existing = container.children;
  while (existing.length >= 3) {
    existing[0].remove();
  }

  const icons = { success: '✅', error: '❌', warning: '⚠️' };
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] || ''}</span><span class="toast-msg">${escapeHtml(message)}</span>`;
  container.appendChild(el);

  // Auto-remove after 4 s
  const timer = setTimeout(() => {
    el.classList.add('toast-out');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    setTimeout(() => { if (el.parentNode) el.remove(); }, 400);
  }, 4000);

  el.addEventListener('click', () => {
    clearTimeout(timer);
    el.classList.add('toast-out');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    setTimeout(() => { if (el.parentNode) el.remove(); }, 400);
  });
}

/** Escape HTML to prevent XSS in toast messages */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

/* ========================================================================
 * 5. API CLIENT
 * ======================================================================== */

const BASE = '/api/v1';

const api = {
  async uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Upload failed (${res.status})`);
    }
    return res.json();
  },

  async listDocuments() {
    const res = await fetch(`${BASE}/documents`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Failed to load documents (${res.status})`);
    }
    return res.json();
  },

  async deleteDocument(docId) {
    const res = await fetch(`${BASE}/documents/${encodeURIComponent(docId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Delete failed (${res.status})`);
    }
    return res.json();
  },

  async askQuestion(question, topK = 5, mode = 'rag', signal = null) {
    const res = await fetch(`${BASE}/qa/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK, mode }),
      signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `QA request failed (${res.status})`);
    }
    return res;
  },

  async compareQuestion(question, topK = 5, signal = null) {
    const res = await fetch(`${BASE}/qa/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK }),
      signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Compare request failed (${res.status})`);
    }
    return res;
  },
};

/* ========================================================================
 * 6. SSE STREAM PARSER
 * ======================================================================== */

async function parseSSEStream(response, callbacks) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Normalize CRLF (\r\n) → LF (\n) — HTTP/SSE uses \r\n line endings
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n').replace(/\r/g, '\n');

      // Split on double-newline (standard SSE event boundary)
      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.trim()) continue;
        await processSSEMessage(part, callbacks);
      }
    }

    if (buffer.trim()) {
      await processSSEMessage(buffer, callbacks);
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    if (callbacks.onError) callbacks.onError(err);
    return;
  } finally {
    reader.releaseLock();
  }
}

async function processSSEMessage(raw, callbacks) {
  const lines = raw.split('\n');
  let eventType = null;
  let dataStr = null;

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      dataStr = line.slice(6);
    }
  }

  if (!eventType || dataStr === null) return;

  // Yield to the browser event loop after answer tokens to force repaint,
  // preventing batched rendering that hides the streaming effect.
  const isAnswerToken = (eventType === 'answer' || eventType === 'rag_answer' || eventType === 'bare_answer');

  try {
    switch (eventType) {
      case 'retrieval': {
        const parsed = JSON.parse(dataStr);
        if (callbacks.onRetrieval) callbacks.onRetrieval(parsed.chunks || []);
        break;
      }
      case 'answer': {
        if (callbacks.onAnswer) callbacks.onAnswer(dataStr);
        break;
      }
      case 'citations': {
        const parsed = JSON.parse(dataStr);
        if (callbacks.onCitations) callbacks.onCitations(parsed);
        break;
      }
      case 'rag_answer': {
        if (callbacks.onRagAnswer) callbacks.onRagAnswer(dataStr);
        break;
      }
      case 'bare_answer': {
        if (callbacks.onBareAnswer) callbacks.onBareAnswer(dataStr);
        break;
      }
      case 'rag_citations': {
        const parsed = JSON.parse(dataStr);
        if (callbacks.onRagCitations) callbacks.onRagCitations(parsed);
        break;
      }
      case 'done': {
        if (callbacks.onDone) callbacks.onDone();
        break;
      }
    }

    // Force browser repaint after each answer token so streaming is visible
    if (isAnswerToken) {
      await new Promise(r => setTimeout(r, 0));
    }
  } catch (err) {
    if (callbacks.onError) callbacks.onError(err);
  }
}

/* ========================================================================
 * 7. TAB & MODE SWITCHING (with hash sync & state persistence)
 * ======================================================================== */

function switchTab(tab, updateHash = true) {
  if (state.tab === tab) return;
  state.tab = tab;
  persistState();

  // Toggle sidebar links
  $$('.sidebar-link').forEach(link => {
    link.classList.toggle('active', link.dataset.tab === tab);
  });

  // Toggle tab content sections
  $$('.tab-content').forEach(section => {
    section.classList.toggle('active', section.id === `tab-${tab}`);
  });

  // Sync URL hash (without creating duplicate history entries)
  if (updateHash) {
    syncHashFromTab();
  }

  // Reload document list when switching to documents tab
  if (tab === 'documents') {
    loadDocuments();
  }
}

function switchMode(mode) {
  if (state.mode === mode) return;

  const prevMode = state.mode;
  state.mode = mode;
  persistState();

  // Toggle mode buttons
  $$('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  const chatMessages = $('#chatMessages');
  const compareContainer = $('#compareQaContainer');
  const citationsArea = $('#citationsArea');

  if (mode === 'compare') {
    // Entering Compare: save current mode's history, switch to compare view
    saveChatBubblesToHistory(prevMode);
    if (chatMessages) chatMessages.classList.add('hidden');
    if (compareContainer) compareContainer.classList.remove('hidden');
    if (citationsArea) citationsArea.classList.add('hidden');
    // Restore compare history if any
    restoreCompareHistory();
  } else {
    // Switching between RAG / bare LLM: swap histories
    if (chatMessages) chatMessages.classList.remove('hidden');
    if (compareContainer) compareContainer.classList.add('hidden');
    if (citationsArea) citationsArea.classList.remove('hidden');

    // Save previous mode's bubbles, then restore current mode's bubbles
    if (prevMode === 'compare') {
      saveCompareHistory();
    } else {
      saveChatBubblesToHistory(prevMode);
    }
    restoreChatBubblesFromHistory(mode);
  }

  updateSendButton();
}

/** Save compare mode pane contents to history */
function saveCompareHistory() {
  const ragContent = document.getElementById('qaRagContent');
  const bareContent = document.getElementById('qaBareContent');
  if (!ragContent || !bareContent) return;

  // Extract text content from each pane (skip empty placeholders)
  const ragText = ragContent.textContent.trim();
  const bareText = bareContent.textContent.trim();

  state.chatHistory.compare = state.chatHistory.compare || [];
  // Only save if both panes have real content (not just placeholder)
  if (ragText && bareText &&
      !ragContent.querySelector('.pane-empty') &&
      !bareContent.querySelector('.pane-empty')) {
    // Check if last compare entry has same content to avoid duplicates
    const lastEntry = state.chatHistory.compare[state.chatHistory.compare.length - 1];
    if (!lastEntry || lastEntry.rag !== ragText || lastEntry.bare !== bareText) {
      state.chatHistory.compare.push({
        question: state.lastCompareQuestion || '',
        rag: ragText,
        bare: bareText,
        ragSources: state.lastCompareRagSources || [],
      });
    }
  }
  persistState();
}

/** Restore compare mode history — show last comparison if available */
function restoreCompareHistory() {
  const history = state.chatHistory.compare || [];
  if (history.length > 0) {
    const last = history[history.length - 1];
    // Restore into panes
    const ragContent = document.getElementById('qaRagContent');
    const bareContent = document.getElementById('qaBareContent');
    if (ragContent && bareContent) {
      ragContent.innerHTML = `<div class="pane-text">${escapeHtml(last.rag)}</div>`;
      bareContent.innerHTML = `<div class="pane-text">${escapeHtml(last.bare)}</div>`;
      // Restore sources badge
      updateCompareSourceBadge(last.ragSources ? last.ragSources.length : 0);
    }
  } else {
    clearComparePanes();
  }
}

/** Serialize current DOM message-bubbles into state.chatHistory[mode]. */
function saveChatBubblesToHistory(mode) {
  const container = $('#chatMessages');
  if (!container || (mode !== 'rag' && mode !== 'bare')) return;
  const bubbles = $$('.message-bubble', container);
  state.chatHistory[mode] = bubbles.map(b => ({
    role: b.classList.contains('message-user') ? 'user' : 'assistant',
    text: (b.querySelector('.message-content') || {}).textContent || '',
    meta: (b.querySelector('.message-meta') || {}).textContent || '',
  }));
  persistState();
}

/** Rebuild DOM from state.chatHistory[mode]. */
function restoreChatBubblesFromHistory(mode) {
  const container = $('#chatMessages');
  if (!container || (mode !== 'rag' && mode !== 'bare')) return;
  // Clear existing bubbles first
  container.querySelectorAll('.message-bubble').forEach(el => el.remove());
  // Restore from history
  const history = state.chatHistory[mode] || [];
  for (const msg of history) {
    addMessageBubble(msg.role, msg.text);
    // Restore meta info if present
    if (msg.meta) {
      const lastBubble = container.querySelector('.message-bubble:last-child .message-meta');
      if (lastBubble) lastBubble.textContent = msg.meta;
    }
  }
  updateChatEmptyState();
}

function clearChatMessages() {
  const container = $('#chatMessages');
  if (!container) return;
  container.querySelectorAll('.message-bubble').forEach(el => el.remove());
  // Also clear the current mode's persisted history
  if (state.mode === 'rag' || state.mode === 'bare') {
    state.chatHistory[state.mode] = [];
    persistState();
  }
  updateChatEmptyState();
  const citationsArea = $('#citationsArea');
  if (citationsArea) citationsArea.classList.add('hidden');
  const citationsList = $('#citationsList');
  if (citationsList) citationsList.innerHTML = '';
}

function clearComparePanes() {
  const ragContent = $('#qaRagContent');
  const bareContent = $('#qaBareContent');
  if (ragContent) {
    ragContent.innerHTML = '<div class="pane-empty">RAG response will stream here</div>';
  }
  if (bareContent) {
    bareContent.innerHTML = '<div class="pane-empty">Bare LLM response will stream here</div>';
  }
}

/** Update the RAG pane's source badge count */
function updateCompareSourceBadge(count) {
  const badge = document.querySelector('.rag-pane .pane-badge');
  if (badge && count > 0) {
    badge.textContent = `${count} sources`;
  }
}

function updateChatEmptyState() {
  const container = $('#chatMessages');
  const emptyState = $('#chatEmpty');
  if (!container || !emptyState) return;
  const hasMessages = container.querySelectorAll('.message-bubble').length > 0;
  emptyState.style.display = hasMessages ? 'none' : '';
}

/* ========================================================================
 * 8. DOCUMENT MANAGEMENT
 * ======================================================================== */

async function loadDocuments() {
  try {
    state.documents = await api.listDocuments();
  } catch (err) {
    toast(err.message, 'error');
    state.documents = [];
  }
  renderDocumentList();
  updateEmptyState();
}

function renderDocumentList() {
  const grid = $('#documentGrid');
  if (!grid) return;
  grid.innerHTML = '';

  if (state.documents.length === 0) return;

  for (const doc of state.documents) {
    const card = document.createElement('div');
    card.className = 'doc-card';

    const fileType = (doc.file_type || '').replace('.', '').toUpperCase();
    const truncatedName = truncate(doc.filename || 'Unknown', 30);
    const chunkCount = doc.chunk_count ?? 0;
    const createdAt = doc.created_at ? formatDate(doc.created_at) : '';

    card.innerHTML = `
      <div class="doc-card-icon">${fileTypeIcon(fileType)}</div>
      <div class="doc-card-info">
        <div class="doc-card-filename" title="${escapeHtml(doc.filename || '')}">${escapeHtml(truncatedName)}</div>
        <div class="doc-card-meta">
          <span class="doc-card-badge">${escapeHtml(fileType)}</span>
          <span class="doc-card-chunks">${chunkCount} chunk${chunkCount !== 1 ? 's' : ''}</span>
          ${createdAt ? `<span class="doc-card-date">${escapeHtml(createdAt)}</span>` : ''}
        </div>
      </div>
      <button class="doc-card-delete" data-doc-id="${escapeHtml(doc.doc_id)}" title="Delete document">&times;</button>
    `;

    const deleteBtn = card.querySelector('.doc-card-delete');
    deleteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      handleDeleteDocument(doc.doc_id, doc.filename);
    });

    grid.appendChild(card);
  }
}

function updateEmptyState() {
  const grid = $('#documentGrid');
  const empty = $('#documentsEmpty');
  const countLabel = $('#docCount');
  if (!grid || !empty) return;

  const hasDocs = state.documents.length > 0;
  empty.style.display = hasDocs ? 'none' : '';
  if (countLabel) {
    countLabel.textContent = `${state.documents.length} document${state.documents.length !== 1 ? 's' : ''}`;
  }
}

async function handleUpload(file) {
  // Validate file type
  const allowedExts = ['.pdf', '.docx', '.md', '.txt'];
  const fileName = file.name.toLowerCase();
  const ext = '.' + fileName.split('.').pop();
  if (!allowedExts.includes(ext)) {
    toast('Unsupported file type. Please upload .pdf, .docx, .md, or .txt files.', 'warning');
    return;
  }

  // Validate file size (max 10 MB)
  const maxSize = 10 * 1024 * 1024;
  if (file.size > maxSize) {
    toast('File is too large. Maximum size is 10MB.', 'warning');
    return;
  }

  // Show progress
  state.isUploading = true;
  const progressEl = $('#uploadProgress');
  const progressFill = $('#progressFill');
  const progressText = $('#progressText');
  const uploadZone = $('#uploadZone');
  if (progressEl) progressEl.classList.remove('hidden');
  if (uploadZone) uploadZone.style.pointerEvents = 'none';
  if (progressFill) progressFill.style.width = '0%';
  if (progressText) {
    progressText.textContent = 'Uploading...';
    progressText.className = 'progress-text uploading';
  }

  let simProgress = 0;
  const simInterval = setInterval(() => {
    simProgress = Math.min(simProgress + Math.random() * 15, 85);
    if (progressFill) progressFill.style.width = simProgress + '%';
  }, 200);

  try {
    await api.uploadDocument(file);
    clearInterval(simInterval);
    if (progressFill) progressFill.style.width = '100%';
    if (progressText) {
      progressText.textContent = '✓ Uploaded successfully';
      progressText.className = 'progress-text success';
    }
    toast(`"${file.name}" uploaded successfully.`, 'success');
    await loadDocuments();

    const fileInput = $('#fileInput');
    if (fileInput) fileInput.value = '';
  } catch (err) {
    clearInterval(simInterval);
    if (progressFill) {
      progressFill.style.width = '100%';
      progressFill.style.background = 'var(--accent-error)';
    }
    if (progressText) {
      progressText.textContent = '✗ Upload failed';
      progressText.className = 'progress-text error';
    }
    toast(err.message, 'error');
  } finally {
    state.isUploading = false;
    // Delayed cleanup — keep status visible for 2s so user can read it
    setTimeout(() => {
      if (progressEl) progressEl.classList.add('hidden');
      if (progressFill) {
        progressFill.style.width = '0%';
        progressFill.style.background = ''; // reset color
      }
      if (progressText) {
        progressText.textContent = 'Uploading...';
        progressText.className = 'progress-text';
      }
      if (uploadZone) uploadZone.style.pointerEvents = '';
    }, 2500);
  }
}

async function handleDeleteDocument(docId, filename) {
  if (!confirm(`Delete "${filename}"?\n\nThis will remove the document and all its chunks from the knowledge base.`)) {
    return;
  }
  try {
    await api.deleteDocument(docId);
    toast(`"${filename}" deleted.`, 'success');
    await loadDocuments();
  } catch (err) {
    toast(err.message, 'error');
  }
}

function fileTypeIcon(type) {
  const map = { PDF: '\u{1F4C4}', DOCX: '\u{1F4DD}', MD: '\u{1F4DD}', TXT: '\u{1F4C3}' };
  return map[type] || '\u{1F4C1}';
}

function truncate(str, max) {
  if (str.length <= max) return str;
  return str.slice(0, max - 1) + '\u2026';
}

function formatDate(isoStr) {
  try {
    const d = new Date(isoStr);
    return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

/* ========================================================================
 * 9. QA CHAT (RAG / Bare modes)
 * ======================================================================== */

async function sendQuestion() {
  const input = $('#questionInput');
  if (!input) return;
  const question = input.value.trim();
  if (!question) return;
  if (state.isStreaming) return;

  abortCurrentStream();

  input.value = '';
  input.style.height = 'auto';

  addMessageBubble('user', question);
  updateChatEmptyState();

  const assistantBubble = addMessageBubble('assistant', '', true);
  const messageContent = assistantBubble.querySelector('.message-content');
  const messageMeta = assistantBubble.querySelector('.message-meta');

  const citationsArea = $('#citationsArea');
  const citationsList = $('#citationsList');
  if (citationsArea) citationsArea.classList.add('hidden');
  if (citationsList) citationsList.innerHTML = '';

  state.isStreaming = true;
  updateSendButton();
  toggleSpinner(true);

  const mode = state.mode;
  state.abortController = new AbortController();

  let retrievalInfoAdded = false;

  try {
    const response = await api.askQuestion(question, 5, mode, state.abortController.signal);

    await parseSSEStream(response, {
      onRetrieval(chunks) {
        if (!retrievalInfoAdded && chunks.length > 0) {
          retrievalInfoAdded = true;
          if (messageMeta) {
            messageMeta.textContent = `Found ${chunks.length} relevant chunk${chunks.length !== 1 ? 's' : ''}`;
          }
        }
      },

      onAnswer(token) {
        if (messageContent) {
          messageContent.textContent += cleanStreamToken(token);
        }
        autoScrollChat();
      },

      onCitations(citations) {
        if (citations && citations.length > 0) {
          // Filter to only show chunks actually referenced in the answer via [N] markers
          const answerText = messageContent ? messageContent.textContent : '';
          const filtered = filterCitationsByReferences(citations, answerText);
          renderCitations(filtered);
        }
      },

      onDone() {
        finishStream(assistantBubble);
      },

      onError(err) {
        console.error('[QA Stream Error]', err);
        if (messageContent) {
          if (!messageContent.textContent.trim()) {
            messageContent.textContent = '⚠️ An error occurred while generating the response.';
          } else {
            messageContent.textContent += '\n\n⚠️ [Stream error: ' + err.message + ']';
          }
        }
        finishStream(assistantBubble);
        toast('Response error: ' + err.message, 'error');
      },
    });
  } catch (err) {
    console.error('[QA Request Error]', err);
    if (err.name !== 'AbortError') {
      if (messageContent && !messageContent.textContent.trim()) {
        messageContent.textContent = '❌ Request failed: ' + err.message + '\n\nPlease check your API key configuration or try again.';
      }
      finishStream(assistantBubble);
      toast('Request failed: ' + err.message, 'error');
    } else {
      assistantBubble.remove();
      updateChatEmptyState();
    }
  }
}

/**
 * Filter citations to only include chunks that are actually referenced
 * in the answer text via [N] markers (e.g. [2], ["3"], etc.)
 * If no references are found, show all citations as fallback.
 */
function filterCitationsByReferences(citations, answerText) {
  if (!citations || citations.length === 0) return [];
  // Extract all [N] reference numbers from the answer text
  const refMatches = answerText.match(/\[["']?(\d+)["']?\]/g);
  if (!refMatches || refMatches.length === 0) {
    // No explicit references found — show all (fallback)
    return citations;
  }
  const referencedIndices = new Set(
    refMatches.map(m => parseInt(m.match(/\d+/)[0], 10))
  );
  // Filter: keep citations whose chunk_index is in the referenced set
  const filtered = citations.filter(c => referencedIndices.has(c.chunk_index));
  // If filtering removed everything (edge case), fall back to all
  return filtered.length > 0 ? filtered : citations;
}

function renderCitations(citations) {
  const citationsArea = $('#citationsArea');
  const citationsList = $('#citationsList');
  if (!citationsArea || !citationsList) return;

  citationsArea.classList.remove('hidden');
  citationsArea.classList.remove('expanded'); // default collapsed
  citationsList.innerHTML = '';
  citationsList.classList.add('collapsed');

  // Update count badge
  const countEl = $('#citationsCount');
  if (countEl) {
    countEl.textContent = citations.length;
  }

  for (const cite of citations) {
    const item = document.createElement('div');
    item.className = 'citation-item';

    const snippet = cite.content_snippet
      ? truncate(cite.content_snippet, 120)
      : 'No preview available';

    item.innerHTML = `
      <div class="citation-header">
        <span class="citation-filename">${escapeHtml(cite.filename || 'Unknown')}</span>
        ${cite.chunk_index != null ? `<span class="citation-chunk">Chunk #${cite.chunk_index}</span>` : ''}
      </div>
      <div class="citation-snippet">${escapeHtml(snippet)}</div>
    `;
    citationsList.appendChild(item);
  }
}

/** Toggle citations panel expand/collapse */
function toggleCitations() {
  const area = $('#citationsArea');
  if (!area) return;
  const isExpanded = area.classList.toggle('expanded');
  const header = $('#citationsHeader');
  if (header) header.setAttribute('aria-expanded', String(isExpanded));
}

function finishStream(bubble) {
  if (bubble) bubble.classList.remove('streaming');
  state.isStreaming = false;
  state.abortController = null;
  updateSendButton();
  toggleSpinner(false);
}

function abortCurrentStream() {
  if (state.abortController) {
    state.abortController.abort();
    state.abortController = null;
  }
}

/* ========================================================================
 * 10. COMPARE VIEW (RAG vs LLM)
 * ======================================================================== */

async function sendCompare() {
  const input = $('#questionInput');
  if (!input) return;
  const question = input.value.trim();
  if (!question) return;
  if (state.isStreaming) return;

  abortCurrentStream();

  input.value = '';
  input.style.height = 'auto';

  addMessageBubble('user', question);
  updateChatEmptyState();

  // Track for compare history persistence
  state.lastCompareQuestion = question;
  state.lastCompareRagSources = [];

  clearComparePanes();
  const ragContent = $('#qaRagContent');
  const bareContent = $('#qaBareContent');

  const citationsArea = $('#citationsArea');
  if (citationsArea) citationsArea.classList.add('hidden');

  state.isStreaming = true;
  updateSendButton();
  toggleSpinner(true);

  state.abortController = new AbortController();

  let ragCitationsReceived = false;

  try {
    const response = await api.compareQuestion(question, 5, state.abortController.signal);

    await parseSSEStream(response, {
      onRagAnswer(token) {
        if (ragContent && ragContent.querySelector('.pane-empty')) {
          ragContent.innerHTML = '';
        }
        if (ragContent) {
          ragContent.appendChild(document.createTextNode(cleanStreamToken(token)));
        }
        autoScrollPane('qaRagContent');
      },

      onBareAnswer(token) {
        if (bareContent && bareContent.querySelector('.pane-empty')) {
          bareContent.innerHTML = '';
        }
        if (bareContent) {
          bareContent.appendChild(document.createTextNode(cleanStreamToken(token)));
        }
        autoScrollPane('qaBareContent');
      },

      onRagCitations(citations) {
        if (!ragCitationsReceived && citations && citations.length > 0) {
          ragCitationsReceived = true;
          state.lastCompareRagSources = citations;
          if (ragContent) {
            const divider = document.createElement('div');
            divider.className = 'pane-citations';
            divider.innerHTML = `<div class="pane-citations-header">\u{1F4D6} Sources (${citations.length})</div>`;
            for (const cite of citations) {
              const item = document.createElement('div');
              item.className = 'pane-citation-item';
              const snippet = cite.content_snippet
                ? truncate(cite.content_snippet, 120)
                : 'No preview';
              item.innerHTML = `
                <span class="pane-citation-file">${escapeHtml(cite.filename || 'Unknown')}</span>
                <span class="pane-citation-snippet">${escapeHtml(snippet)}</span>
              `;
              divider.appendChild(item);
            }
            ragContent.appendChild(divider);
          }
        }
      },

      onDone() {
        state.isStreaming = false;
        state.abortController = null;
        updateSendButton();
        toggleSpinner(false);
        if (ragContent && ragContent.querySelector('.pane-empty') && ragContent.textContent.trim() === 'RAG response will stream here') {
          ragContent.querySelector('.pane-empty').textContent = '(No answer received)';
        }
        if (bareContent && bareContent.querySelector('.pane-empty') && bareContent.textContent.trim() === 'Bare LLM response will stream here') {
          bareContent.querySelector('.pane-empty').textContent = '(No answer received)';
        }
      },

      onError(err) {
        state.isStreaming = false;
        state.abortController = null;
        updateSendButton();
        toggleSpinner(false);
        if (ragContent) {
          ragContent.innerHTML = `<div class="pane-error">Error: ${escapeHtml(err.message)}</div>`;
        }
        if (bareContent) {
          bareContent.innerHTML = `<div class="pane-error">Error: ${escapeHtml(err.message)}</div>`;
        }
        toast('Compare stream error: ' + err.message, 'error');
      },
    });
  } catch (err) {
    if (err.name !== 'AbortError') {
      if (ragContent) {
        ragContent.innerHTML = `<div class="pane-error">Request failed: ${escapeHtml(err.message)}</div>`;
      }
      if (bareContent) {
        bareContent.innerHTML = `<div class="pane-error">Request failed: ${escapeHtml(err.message)}</div>`;
      }
      toast(err.message, 'error');
    }
    state.isStreaming = false;
    state.abortController = null;
    updateSendButton();
    toggleSpinner(false);
  }
}

/* ========================================================================
 * 11. STREAM OUTPUT CLEANER
 * ======================================================================== */

/**
 * Clean raw LLM stream tokens before displaying in chat bubbles.
 * Removes technical markers that leak from the model's raw output.
 */
function cleanStreamToken(raw) {
  if (!raw) return '';
  let cleaned = raw;
  // Remove surrounding double quotes (SSE data wraps tokens in "")
  cleaned = cleaned.replace(/^"|"$/g, '');
  // Convert literal \n\n (escaped newlines) → actual paragraph breaks
  cleaned = cleaned.replace(/\\n\\n/g, '\n\n');
  // Convert literal \n (escaped newline) → space
  cleaned = cleaned.replace(/\\n/g, ' ');
  // Remove citation markers like [1], ["2"], [3], etc.
  cleaned = cleaned.replace(/\["?\d*"\]?/g, '');
  // Remove stray double-quote pairs used as word/phrase separators: "word" → word
  cleaned = cleaned.replace(/"([^"]*)"/g, '$1');
  // Collapse multiple spaces into one
  cleaned = cleaned.replace(/ {2,}/g, ' ');
  // Trim leading/trailing whitespace on each line
  cleaned = cleaned.split('\n').map(line => line.trim()).join('\n');
  return cleaned;
}

/* ========================================================================
 * 12. CHAT / COMPARE UTILITY HELPERS
 * ======================================================================== */

function addMessageBubble(role, text, streaming = false) {
  const container = $('#chatMessages');
  if (!container) return document.createElement('div');

  const bubble = document.createElement('div');
  bubble.className = `message-bubble message-${role}`;
  if (streaming) bubble.classList.add('streaming');

  const roleLabel = role === 'user' ? 'You' : 'AI';
  const roleClass = role === 'user' ? 'user-label' : 'ai-label';

  bubble.innerHTML = `
    <div class="message-role ${roleClass}">${roleLabel}</div>
    <div class="message-content">${escapeHtml(text)}</div>
    <div class="message-meta"></div>
  `;

  container.appendChild(bubble);
  autoScrollChat();
  return bubble;
}

function autoScrollChat() {
  const el = $('#chatMessages');
  if (el) el.scrollTop = el.scrollHeight;
}

function autoScrollPane(paneId) {
  const el = document.getElementById(paneId);
  if (el) el.scrollTop = el.scrollHeight;
}

function updateSendButton() {
  const btn = $('#btnSendQa');
  const input = $('#questionInput');
  if (!btn) return;
  const hasText = input ? input.value.trim().length > 0 : false;
  btn.disabled = !hasText || state.isStreaming;
}

function toggleSpinner(show) {
  const spinner = $('#qaSpinner');
  const icon = $('#btnSendQa') ? $('#btnSendQa').querySelector('.send-icon') : null;
  if (spinner) spinner.classList.toggle('hidden', !show);
  if (icon) icon.style.display = show ? 'none' : '';
}

/* ========================================================================
 * 12. EVENT BINDINGS & INIT
 * ======================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // ---- Restore persisted tab & mode ----
  const initialHash = window.location.hash.replace('#', '');
  if (initialHash === 'qa') {
    state.tab = 'qa';
  } else if (initialHash === 'documents') {
    state.tab = 'documents';
  }
  // Else keep persisted state

  // Apply initial tab state to UI
  $$('.sidebar-link').forEach(link => {
    link.classList.toggle('active', link.dataset.tab === state.tab);
  });
  $$('.tab-content').forEach(section => {
    section.classList.toggle('active', section.id === `tab-${state.tab}`);
  });

  // Apply initial mode state to UI
  $$('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === state.mode);
  });

  // Apply compare view if mode is 'compare'
  if (state.mode === 'compare') {
    const chatMessages = $('#chatMessages');
    const compareContainer = $('#compareQaContainer');
    const citationsArea = $('#citationsArea');
    if (chatMessages) chatMessages.classList.add('hidden');
    if (compareContainer) compareContainer.classList.remove('hidden');
    if (citationsArea) citationsArea.classList.add('hidden');
  } else {
    // Restore per-mode chat history on page load
    restoreChatBubblesFromHistory(state.mode);
  }

  // ---- Sidebar navigation ----
  $$('.sidebar-link').forEach(link => {
    link.addEventListener('click', () => switchTab(link.dataset.tab, true));
  });

  // ---- Mode pills ----
  $$('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => switchMode(btn.dataset.mode));
  });

  // ---- Upload zone: click to open file picker ----
  const uploadZone = $('#uploadZone');
  const fileInput = $('#fileInput');
  if (uploadZone && fileInput) {
    uploadZone.addEventListener('click', (e) => {
      if (e.target !== fileInput) fileInput.click();
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files[0]) handleUpload(e.target.files[0]);
    });

    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('dragover');
    });
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
      if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
    });
  }

  // ---- Send button ----
  const btnSendQa = $('#btnSendQa');
  if (btnSendQa) {
    btnSendQa.addEventListener('click', () => {
      if (state.mode === 'compare') {
        sendCompare();
      } else {
        sendQuestion();
      }
    });
  }

  // ---- Citations panel toggle ----
  const citationsHeader = $('#citationsHeader');
  if (citationsHeader) {
    const toggleHandler = () => toggleCitations();
    citationsHeader.addEventListener('click', toggleHandler);
    citationsHeader.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        toggleHandler();
      }
    });
  }

  // ---- Question input ----
  const questionInput = $('#questionInput');
  if (questionInput) {
    questionInput.addEventListener('input', () => {
      questionInput.style.height = 'auto';
      questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
      updateSendButton();
    });

    questionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!state.isStreaming && questionInput.value.trim()) {
          if (state.mode === 'compare') {
            sendCompare();
          } else {
            sendQuestion();
          }
        }
      }
    });
  }

  // ---- Initial data load ----
  loadDocuments();

  // ---- Sync initial hash ----
  syncHashFromTab();
});
