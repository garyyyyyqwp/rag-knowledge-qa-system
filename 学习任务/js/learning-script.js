/**
 * RAG 学习路径 — 共享交互脚本
 * 功能: 代码复制、折叠面板、进度条
 */

document.addEventListener('DOMContentLoaded', function () {
  initCopyButtons();
  initCollapsibles();
  initProgressBar();
});

// --- 代码复制按钮 ---
function initCopyButtons() {
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var codeBlock = btn.closest('.code-block');
      var pre = codeBlock.querySelector('pre');
      var text = pre ? pre.textContent : '';

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          showCopied(btn);
        }).catch(function () {
          fallbackCopy(text, btn);
        });
      } else {
        fallbackCopy(text, btn);
      }
    });
  });
}

function fallbackCopy(text, btn) {
  var textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand('copy');
    showCopied(btn);
  } catch (e) {
    btn.textContent = '复制失败';
  }
  document.body.removeChild(textarea);
}

function showCopied(btn) {
  var original = btn.textContent;
  btn.textContent = '已复制!';
  btn.classList.add('copied');
  setTimeout(function () {
    btn.textContent = original;
    btn.classList.remove('copied');
  }, 2000);
}

// --- 折叠面板 ---
function initCollapsibles() {
  document.querySelectorAll('.collapsible-header').forEach(function (header) {
    header.addEventListener('click', function () {
      var parent = header.parentElement;
      parent.classList.toggle('open');
    });
  });
}

// --- 进度条 ---
function initProgressBar() {
  var fill = document.querySelector('.progress-bar .fill');
  if (!fill) return;

  var currentNum = parseInt(fill.getAttribute('data-current') || '1');
  var totalNum = parseInt(fill.getAttribute('data-total') || '12');
  var percent = Math.round((currentNum / totalNum) * 100);
  fill.style.width = percent + '%';
}
