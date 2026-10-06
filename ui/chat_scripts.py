
GLOBAL_NAV_ARROWS_HTML = """
<script>
    const parentDoc = window.parent.document;
    if (!parentDoc.getElementById('nav-arrows')) {
        const container = parentDoc.createElement('div');
        container.id = 'nav-arrows';
        container.innerHTML = `
            <style>
            .nav-arrows-container {
                position: fixed;
                right: 32px;
                top: 50%;
                transform: translateY(-50%);
                display: flex;
                flex-direction: column;
                gap: 12px;
                z-index: 9999;
            }
            .nav-arrow-btn {
                background: rgba(10, 10, 10, 0.6);
                border: 1px solid rgba(143, 138, 130, 0.3);
                color: #8f8a82;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
            }
            .nav-arrow-btn:hover {
                transform: scale(1.15);
                background: rgba(30, 30, 30, 0.9);
                color: #eab308;
                border-color: #eab308;
            }
            .copy-btn {
                cursor: pointer;
                transition: transform 0.1s ease;
                display: flex;
                align-items: center;
            }
            .copy-btn:hover {
                color: #fff;
                transform: scale(1.15);
            }
            .user-request-anchor {
                scroll-margin-top: 25vh;
                display: block;
                height: 1px;
            }
            </style>
            <div class="nav-arrows-container">
                <div class="nav-arrow-btn" id="nav-up" title="Previous Request">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 15l-6-6-6 6"/></svg>
                </div>
                <div class="nav-arrow-btn" id="nav-down" title="Next Request">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
                </div>
            </div>
        `;
        parentDoc.body.appendChild(container);
        
        let currentScrollIndex = -1;

        function updateArrowsVisibility() {
            const reqs = Array.from(parentDoc.querySelectorAll('.user-request-anchor'));
            const upBtn = parentDoc.getElementById('nav-up');
            const downBtn = parentDoc.getElementById('nav-down');
            const container = parentDoc.querySelector('.nav-arrows-container');
            const navWrapper = parentDoc.getElementById('nav-arrows');
            
            if (!container) return;
            
            if (reqs.length <= 1) {
                container.style.display = 'none';
                if (navWrapper) navWrapper.style.display = 'none';
                return;
            }
            container.style.display = 'flex';
            if (navWrapper) navWrapper.style.display = 'block';
            
            if (currentScrollIndex <= 0) {
                upBtn.style.opacity = '0.3';
                upBtn.style.pointerEvents = 'none';
            } else {
                upBtn.style.opacity = '1';
                upBtn.style.pointerEvents = 'auto';
            }
            
            if (currentScrollIndex >= reqs.length - 1) {
                downBtn.style.opacity = '0.3';
                downBtn.style.pointerEvents = 'none';
            } else {
                downBtn.style.opacity = '1';
                downBtn.style.pointerEvents = 'auto';
            }
        }

        const existingReqs = Array.from(parentDoc.querySelectorAll('.user-request-anchor'));
        if (existingReqs.length > 0) {
            currentScrollIndex = existingReqs.length - 1;
        }

        parentDoc.getElementById('nav-up').onclick = () => {
            const reqs = Array.from(parentDoc.querySelectorAll('.user-request-anchor'));
            if (reqs.length === 0) return;
            if (currentScrollIndex > 0) {
                currentScrollIndex--;
            } else {
                currentScrollIndex = 0;
            }
            
            reqs[currentScrollIndex].scrollIntoView({ behavior: 'smooth', block: 'start' });
            updateArrowsVisibility();
        };
        
        parentDoc.getElementById('nav-down').onclick = () => {
            const reqs = Array.from(parentDoc.querySelectorAll('.user-request-anchor'));
            if (reqs.length === 0) return;
            if (currentScrollIndex < reqs.length - 1) {
                currentScrollIndex++;
            } else {
                currentScrollIndex = reqs.length - 1;
            }
            
            reqs[currentScrollIndex].scrollIntoView({ behavior: 'smooth', block: 'start' });
            updateArrowsVisibility();
        };

        function getScrollContainer() {
            const mainSection = parentDoc.querySelector('section[data-testid="stMain"]') || parentDoc.querySelector('.main');
            if (mainSection && mainSection.scrollHeight > mainSection.clientHeight + 2) {
                return mainSection;
            }
            return parentDoc.scrollingElement || parentDoc.documentElement || parentDoc.body || window.parent;
        }

        let isNearBottom = true;
        let isProgrammaticScroll = false;
        let scrollRafId = null;

        function updateNearBottomState() {
            if (isProgrammaticScroll) return;
            const c = getScrollContainer();
            const scrollHeight = c.scrollHeight || parentDoc.documentElement.scrollHeight || 0;
            const scrollTop = c.scrollTop || parentDoc.documentElement.scrollTop || window.parent.scrollY || 0;
            const clientHeight = c.clientHeight || parentDoc.documentElement.clientHeight || window.parent.innerHeight || 0;
            
            const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
            isNearBottom = distanceFromBottom <= 140;
        }

        const scrollContainer = getScrollContainer();
        if (scrollContainer && scrollContainer.addEventListener) {
            scrollContainer.addEventListener('scroll', updateNearBottomState, { passive: true });
        }
        window.parent.addEventListener('scroll', updateNearBottomState, { passive: true });
        parentDoc.addEventListener('scroll', updateNearBottomState, { passive: true });

        function smartAutoScroll() {
            if (!isNearBottom) return;
            if (scrollRafId) return;

            scrollRafId = requestAnimationFrame(() => {
                scrollRafId = null;
                const c = getScrollContainer();
                const distanceFromBottom = c.scrollHeight - (c.scrollTop + c.clientHeight);
                if (distanceFromBottom > 180) {
                    isNearBottom = false;
                    return;
                }
                isProgrammaticScroll = true;
                if (c.scrollTo) {
                    c.scrollTo({ top: c.scrollHeight, behavior: 'instant' });
                } else if (window.parent.scrollTo) {
                    window.parent.scrollTo({ top: c.scrollHeight, behavior: 'instant' });
                }
                setTimeout(() => { isProgrammaticScroll = false; }, 40);
            });
        }

        function mutationTouchesLiveResponse(records) {
            const responseSelector = '.agent-response-wrapper';
            return records.some(record => {
                if (record.target && record.target.nodeType === 1 && record.target.closest(responseSelector)) return true;
                return Array.from(record.addedNodes || []).some(node => {
                    if (node.nodeType !== 1) return false;
                    return node.matches(responseSelector) || !!node.querySelector(responseSelector) || !!node.closest(responseSelector);
                });
            });
        }

        function fitChatTitleInput(input) {
            if (!input) return;
            const text = input.value || input.placeholder || '';
            const fitKey = `${text}|${input.clientWidth}|${parentDoc.fonts ? parentDoc.fonts.status : 'ready'}`;
            if (input.dataset.ragTitleFitKey === fitKey) return;
            const style = parentDoc.defaultView.getComputedStyle(input);
            const canvas = parentDoc.createElement('canvas');
            const context = canvas.getContext('2d');
            if (!context) return;
            const padding = Number.parseFloat(style.paddingLeft || '0') + Number.parseFloat(style.paddingRight || '0');
            const available = Math.max(80, input.clientWidth - padding - 4);
            const maxSize = 26;
            const minSize = 8;
            const baseLetterSpacing = Number.parseFloat(style.letterSpacing || '0') || 0;
            let size = maxSize;
            while (size >= minSize) {
                context.font = `${style.fontWeight} ${size}px ${style.fontFamily}`;
                const measured = context.measureText(text).width + Math.max(0, text.length - 1) * baseLetterSpacing * (size / maxSize);
                if (measured <= available) break;
                size -= 0.25;
            }
            size = Math.max(minSize, size);
            input.style.setProperty('font-size', `${size}px`, 'important');
            input.scrollLeft = 0;
            input.title = input.value || '';
            input.dataset.ragTitleFitKey = fitKey;
        }

        function positionStopBesideSubmit() {
            const wrapper = parentDoc.querySelector('.st-key-stop_generating_btn');
            const submit = parentDoc.querySelector('[data-testid="stChatInputSubmitButton"]');
            const inputRoot = parentDoc.querySelector('[data-testid="stChatInput"]');
            const activeWork = parentDoc.querySelector('.rag-working-counter[data-working-start]:not([data-working-final])');
            if (!wrapper) return;
            if (!activeWork || !submit || !inputRoot) {
                const originalContainer = wrapper.__ragOriginalContainer;
                wrapper.remove();
                if (originalContainer && originalContainer.isConnected) {
                    originalContainer.style.setProperty('display', 'none', 'important');
                }
                return;
            }
            const originalContainer = wrapper.closest('[data-testid="stElementContainer"]');
            if (!inputRoot.contains(wrapper)) {
                wrapper.__ragOriginalContainer = originalContainer;
                inputRoot.appendChild(wrapper);
                if (originalContainer && !inputRoot.contains(originalContainer)) {
                    originalContainer.style.setProperty('display', 'none', 'important');
                }
            }
            if (parentDoc.defaultView.getComputedStyle(inputRoot).position === 'static') {
                inputRoot.style.setProperty('position', 'relative', 'important');
            }
            if (inputRoot.style.getPropertyValue('overflow') !== 'visible') {
                inputRoot.style.setProperty('overflow', 'visible', 'important');
            }
            const rect = submit.getBoundingClientRect();
            const rootRect = inputRoot.getBoundingClientRect();
            if (!rect.width || !rect.height) return;
            const width = Math.round(rect.width);
            const height = Math.round(rect.height);
            const left = Math.round(rect.left - rootRect.left - width - 8);
            const top = Math.round(rect.top - rootRect.top);
            const setIfChanged = (property, value) => {
                if (wrapper.style.getPropertyValue(property) !== value || wrapper.style.getPropertyPriority(property) !== 'important') {
                    wrapper.style.setProperty(property, value, 'important');
                }
            };
            setIfChanged('display', 'flex');
            setIfChanged('align-items', 'center');
            setIfChanged('justify-content', 'center');
            setIfChanged('position', 'absolute');
            setIfChanged('left', `${left}px`);
            setIfChanged('top', `${top}px`);
            setIfChanged('width', `${width}px`);
            setIfChanged('height', `${height}px`);
            setIfChanged('min-width', `${width}px`);
            setIfChanged('min-height', `${height}px`);
            setIfChanged('box-sizing', 'border-box');
            setIfChanged('visibility', 'visible');

            const stopButton = wrapper.querySelector('button');
            if (stopButton) {
                stopButton.setAttribute('aria-label', 'Stop generating');
                for (const [property, value] of Object.entries({
                    display: 'flex',
                    'align-items': 'center',
                    'justify-content': 'center',
                    width: `${width}px`,
                    'min-width': `${width}px`,
                    height: `${height}px`,
                    'min-height': `${height}px`,
                    padding: '0',
                    overflow: 'hidden',
                    'box-sizing': 'border-box',
                })) {
                    stopButton.style.setProperty(property, value, 'important');
                }
            }
        }

        parentDoc.addEventListener('input', event => {
            if (event.target && event.target.matches('input[aria-label="Chat Title"]')) {
                fitChatTitleInput(event.target);
            }
        }, true);
        window.parent.setInterval(() => {
            fitChatTitleInput(parentDoc.querySelector('input[aria-label="Chat Title"]'));
        }, 300);
        window.parent.addEventListener('resize', () => {
            fitChatTitleInput(parentDoc.querySelector('input[aria-label="Chat Title"]'));
            positionStopBesideSubmit();
        }, { passive: true });

        const observer = new MutationObserver((records) => {
            const reqs = Array.from(parentDoc.querySelectorAll('.user-request-anchor'));
            if (reqs.length > 0 && currentScrollIndex === -1) {
                currentScrollIndex = reqs.length - 1;
            }
            updateArrowsVisibility();

            parentDoc.querySelectorAll('.rag-trace-collapse-after:not([data-collapse-scheduled])').forEach(trace => {
                trace.dataset.collapseScheduled = 'true';
                const delay = parentDoc.defaultView.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 480;
                parentDoc.defaultView.setTimeout(() => {
                    if (!trace.isConnected) return;
                    trace.removeAttribute('open');
                    trace.classList.remove('rag-trace-collapse-after');
                }, delay);
            });
            
            const stopBtn = parentDoc.querySelector('.st-key-stop_generating_btn button');
            const activeWork = parentDoc.querySelector('.rag-working-counter[data-working-start]:not([data-working-final])');
            if (stopBtn) positionStopBesideSubmit();
            const chatSubmit = parentDoc.querySelector('[data-testid="stChatInputSubmitButton"]');
            if (chatSubmit) {
                if (activeWork) {
                    chatSubmit.disabled = true;
                    chatSubmit.style.opacity = '0.5';
                    chatSubmit.style.cursor = 'not-allowed';
                } else {
                    chatSubmit.disabled = false;
                    chatSubmit.style.opacity = '1';
                    chatSubmit.style.cursor = 'pointer';
                }
            }
            const titleInput = parentDoc.querySelector('input[aria-label="Chat Title"]');
            if (titleInput) fitChatTitleInput(titleInput);

            updateWorkingCounters();
            if (stopBtn && mutationTouchesLiveResponse(records)) {
                smartAutoScroll();
            }
        });
        observer.observe(parentDoc.body, { childList: true, subtree: true });

        function formatWorkingTime(totalSeconds) {
            const seconds = Math.max(0, Math.floor(totalSeconds));
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const remainder = seconds % 60;
            if (hours > 0) return `${hours}h ${minutes}m ${remainder}s`;
            if (minutes > 0) return `${minutes}m ${remainder}s`;
            return `${remainder}s`;
        }

        function updateWorkingCounters() {
            parentDoc.querySelectorAll('.rag-working-counter[data-working-start]').forEach(counter => {
                if (counter.dataset.workingFinal) return;
                const startedAt = Number(counter.dataset.workingStart);
                if (!Number.isFinite(startedAt) || startedAt <= 0) return;
                const label = `Working for ${formatWorkingTime((Date.now() - startedAt) / 1000)}`;
                if (counter.textContent !== label) counter.textContent = label;
            });
        }
        updateWorkingCounters();
        window.setInterval(updateWorkingCounters, 1000);
        
        updateArrowsVisibility();
        window.addEventListener('beforeunload', () => {
            const el = parentDoc.getElementById('nav-arrows');
            if (el) el.remove();
        });
        window.addEventListener('pagehide', () => {
            const el = parentDoc.getElementById('nav-arrows');
            if (el) el.remove();
        });
        
        function fallbackCopy(text, doc) {
            return new Promise((resolve, reject) => {
                const targetDoc = doc || document;
                const textArea = targetDoc.createElement("textarea");
                textArea.value = text;
                textArea.style.position = "fixed";
                textArea.style.top = "0";
                textArea.style.left = "0";
                textArea.style.width = "2em";
                textArea.style.height = "2em";
                textArea.style.padding = "0";
                textArea.style.border = "none";
                textArea.style.outline = "none";
                textArea.style.boxShadow = "none";
                textArea.style.background = "transparent";
                textArea.setAttribute("readonly", "");
                targetDoc.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                textArea.setSelectionRange(0, textArea.value.length);
                try {
                    const ok = targetDoc.execCommand('copy');
                    targetDoc.body.removeChild(textArea);
                    if (ok) resolve(); else reject(new Error('execCommand returned false'));
                } catch (err) {
                    targetDoc.body.removeChild(textArea);
                    reject(err);
                }
            });
        }

        const copyHandler = function(btn) {
            if (!btn) return;
            const targetDoc = btn.ownerDocument || parentDoc;
            let el = btn;
            while(el && !el.classList.contains('agent-response-wrapper')) {
                el = el.parentElement;
            }
            if (el) {
                const textElement = el.querySelector('.agent-response-text');
                if (textElement) {
                    const clone = textElement.cloneNode(true);
                    const unwanted = clone.querySelectorAll('.thought-accordion, details, .rag-sources-accordion, .rag-sources-container, .rag-telemetry-container, [data-no-copy="true"]');
                    unwanted.forEach(a => a.remove());
                    const text = (clone.innerText || textElement.innerText).trim();
                    
                    const showSuccess = () => {
                        const original = btn.innerHTML;
                        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
                        btn.style.borderColor = 'rgba(34, 197, 94, 0.6)';
                        btn.style.backgroundColor = 'rgba(34, 197, 94, 0.15)';
                        setTimeout(() => { 
                            btn.innerHTML = original; 
                            btn.style.borderColor = ''; 
                            btn.style.backgroundColor = '';
                        }, 2000);
                    };

                    const targetWin = targetDoc.defaultView || window;
                    if (targetWin.navigator.clipboard && targetWin.isSecureContext) {
                        targetWin.navigator.clipboard.writeText(text).then(showSuccess).catch(() => {
                            fallbackCopy(text, targetDoc).then(showSuccess).catch(console.error);
                        });
                    } else {
                        fallbackCopy(text, targetDoc).then(showSuccess).catch(() => {
                            try {
                                const range = targetDoc.createRange();
                                range.selectNodeContents(textElement);
                                const sel = targetWin.getSelection();
                                sel.removeAllRanges();
                                sel.addRange(range);
                                targetDoc.execCommand('copy');
                                sel.removeAllRanges();
                                showSuccess();
                            } catch(e) {
                                console.error('Final copy fallback failed', e);
                            }
                        });
                    }
                }
            }
        };

        parentDoc.defaultView.copyAgentText = copyHandler;
        window.copyAgentText = copyHandler;

        parentDoc.addEventListener('click', function(e) {
            var btn = e.target.closest('.copy-btn');
            if (btn) {
                e.preventDefault();
                e.stopPropagation();
                copyHandler(btn);
            }
        });
    }
</script>
"""

USER_CARD_HTML = """
<style>
@keyframes morph {
    0%, 100% { border-radius: 40% 60% 70% 30% / 40% 40% 60% 50%; }
    34% { border-radius: 70% 30% 50% 50% / 30% 30% 70% 70%; }
    67% { border-radius: 100% 60% 60% 100% / 100% 100% 60% 60%; }
}
.blob-avatar {
    background-color: #eab308;
    border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
    animation: morph 8s ease-in-out infinite;
    width: 32px;
    height: 32px;
    flex-shrink: 0;
    margin-left: 16px;
}
</style>
<div id="card" class="user-request-card" style="position: relative; padding: 32px 48px; border-radius: 24px; font-family: sans-serif; line-height: 1.5; box-sizing: border-box; overflow: hidden; background-color: rgba(10, 10, 10, 0.85); border: 1px solid rgba(143, 138, 130, 0.5); margin-left: 15%;">
    <div id="content-container" style="position: relative; z-index: 1; display: flex; flex-direction: row-reverse; gap: 16px; align-items: flex-start; text-align: left;">
        <div class="blob-avatar"></div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; width: 100%;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #8f8a82; margin-bottom: 8px; opacity: 0.8; margin-top: 6px;">User Request</div>
            <div id="attachments" style="display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; margin-bottom: 8px; width: 100%;"></div>
            <div id="content" style="white-space: pre-wrap; color: var(--st-text-color); font-size: 18px; text-align: left;"></div>
        </div>
    </div>
</div>
"""

USER_CARD_JS = """
export default function (component) {
  const { data, parentElement } = component;
  const content = parentElement.querySelector("#content");
  const attachments = parentElement.querySelector("#attachments");
  
  if (content) {
      content.textContent = data.text;
  }
  
  if (attachments) {
      attachments.innerHTML = "";
      if (data.files && data.files.length > 0) {
          data.files.forEach(f => {
              const pill = document.createElement("div");
              pill.style.cssText = "display: inline-flex; align-items: center; background: rgba(143, 138, 130, 0.15); padding: 6px 12px; border-radius: 12px; font-size: 13px; color: #d6d3cd; border: 1px solid rgba(143, 138, 130, 0.3); box-shadow: 0 2px 5px rgba(0,0,0,0.2);";
              pill.innerHTML = `<span style="margin-right:6px; opacity:0.8;">📄</span> ${f}`;
              attachments.appendChild(pill);
          });
      }
  }
}
"""

AGENT_RESPONSE_PREFIX = """
<style>
@keyframes morph-strong {
    0%, 100% { border-radius: 40% 60% 70% 30% / 40% 40% 60% 50%; transform: rotate(0deg) scale(1); }
    34% { border-radius: 70% 30% 50% 50% / 30% 30% 70% 70%; transform: rotate(15deg) scale(1.05); }
    67% { border-radius: 100% 60% 60% 100% / 100% 100% 60% 60%; transform: rotate(-10deg) scale(0.95); }
}
.agent-blob {
    background-color: #e11d48;
    animation: morph-strong 5s ease-in-out infinite;
    width: 32px;
    height: 32px;
    flex-shrink: 0;
    margin-top: 4px;
}

.animated-dots {
    display: inline-flex !important;
    align-items: center !important;
    margin-left: 3px !important;
    letter-spacing: 2px !important;
}

.animated-dots .dot {
    opacity: 0.2 !important;
    animation: seqDotPulse 1.4s infinite ease-in-out !important;
    font-weight: 700 !important;
}

.animated-dots .dot-1 {
    animation-delay: 0s !important;
}

.animated-dots .dot-2 {
    animation-delay: 0.25s !important;
}

.animated-dots .dot-3 {
    animation-delay: 0.5s !important;
}

@keyframes seqDotPulse {
    0%, 20% {
        opacity: 0.2;
        transform: translateY(0);
    }
    40% {
        opacity: 1;
        transform: translateY(-1px);
    }
    60%, 100% {
        opacity: 0.2;
        transform: translateY(0);
    }
}

.ragscope-brand-anchor {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 17.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    color: #f4f4f5 !important;
    margin-bottom: 12px !important;
    margin-top: 4px !important;
    line-height: 1.2 !important;
    display: inline-flex !important;
    align-items: center !important;
}
.rag-working-counter {
    margin-left: 12px !important;
    color: #929292 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-variant-numeric: tabular-nums !important;
    white-space: nowrap !important;
}
.rag-trace-details {
    margin: 0 0 14px 0 !important;
    padding: 12px 16px !important;
    border: 1px solid rgba(143, 138, 130, 0.28) !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, rgba(255,255,255,.035), rgba(255,255,255,.012)) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.035) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
}
.rag-trace-live {
    transform-origin: top left !important;
    animation: ragTraceArrive .42s cubic-bezier(.2,.75,.25,1) both !important;
}
.rag-trace-live .rag-process-step,
.rag-trace-live .rag-process-active {
    animation: ragTraceStepIn .38s cubic-bezier(.2,.75,.25,1) both !important;
    animation-delay: calc(var(--trace-index, 0) * 34ms) !important;
}
.rag-trace-streaming .rag-process-step,
.rag-trace-streaming .rag-process-active {
    animation: ragProcessFloat 5s ease-in-out infinite !important;
    animation-delay: calc(var(--trace-index, 0) * 115ms) !important;
}
.rag-trace-streaming .rag-process-dot {
    animation: ragProcessPulse 1.1s ease-in-out infinite alternate !important;
}
.rag-trace-final {
    transform-origin: top left !important;
    animation: ragTraceCollapse .48s cubic-bezier(.2,.75,.25,1) both !important;
}
.rag-trace-collapse-after .rag-trace-content {
    overflow: hidden !important;
    animation: ragTraceContentCollapse .46s cubic-bezier(.4,0,.2,1) both !important;
}
@keyframes ragTraceArrive {
    from { opacity: 0; transform: translateX(10px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes ragTraceStepIn {
    0% { opacity: 0; transform: translateX(14px); }
    70% { opacity: 1; transform: translateX(-3px); }
    100% { opacity: 1; transform: translateX(0); }
}
@keyframes ragTraceCollapse {
    0% { opacity: .72; transform: translateY(-3px) scaleY(1.025); }
    100% { opacity: 1; transform: translateY(0) scaleY(1); }
}
@keyframes ragTraceContentCollapse {
    from { max-height: 900px; opacity: 1; }
    to { max-height: 0; opacity: 0; }
}
@keyframes ragProcessFloat {
    0%, 100% { transform: translateX(0); }
    50% { transform: translateX(0.55px); }
}
@keyframes ragProcessPulse {
    from { opacity: .45; transform: scale(.82); }
    to { opacity: 1; transform: scale(1.12); }
}
.rag-trace-summary {
    display: inline-flex !important;
    align-items: center !important;
    gap: 9px !important;
    cursor: pointer !important;
    list-style: none !important;
    color: #a1a1aa !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    letter-spacing: .01em !important;
    user-select: none !important;
    transition: color .2s ease, transform .35s ease !important;
}
.rag-trace-details:hover .rag-trace-summary { transform: translateX(1px) !important; }
.rag-trace-summary::-webkit-details-marker { display: none !important; }
.rag-trace-chevron {
    width: 7px !important;
    height: 7px !important;
    border-right: 1.5px solid currentColor !important;
    border-bottom: 1.5px solid currentColor !important;
    transform: rotate(-45deg) !important;
    transition: transform .16s ease !important;
}
.rag-trace-details[open] .rag-trace-chevron { transform: rotate(45deg) !important; }
.rag-trace-content {
    padding: 10px 0 2px 16px !important;
    color: #a1a1aa !important;
    font: 14px/1.65 -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
}
.rag-pipeline-steps {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 6px !important;
    margin: 0 0 11px 0 !important;
}
.rag-pipeline-caption {
    margin-right: 3px !important;
    color: #85858d !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: .09em !important;
}
.rag-pipeline-step {
    padding: 3px 8px !important;
    border: 1px solid rgba(143, 138, 130, .24) !important;
    border-radius: 999px !important;
    background: rgba(255, 255, 255, .035) !important;
    color: #d4d4d8 !important;
    font: 11px/1.3 -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    white-space: nowrap !important;
}
.rag-pipeline-arrow {
    color: #777780 !important;
    font: 14px/1 -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
}
.rag-process-step { color: #a1a1aa !important; font: 15px/1.55 -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important; }
.rag-process-active { display: flex !important; align-items: center !important; gap: 9px !important; color: #f1f5f9 !important; font-size: 15px !important; font-weight: 650 !important; line-height: 1.6 !important; }
.rag-process-dot { color: #fda4af !important; font-size: 10px !important; }
.post-generation-item {
    animation: postGenerationEnter .34s cubic-bezier(.2,.75,.25,1) both !important;
}
.post-generation-sources { animation-delay: 0ms !important; }
.post-generation-telemetry { animation-delay: 70ms !important; }
@keyframes postGenerationEnter {
    from { opacity: 0; transform: translateY(-7px); }
    to { opacity: 1; transform: translateY(0); }
}

.fin-num-badge {
    display: inline-block !important;
    padding: 1px 6px !important;
    margin: 0 1px !important;
    border-radius: 5px !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.24) !important;
    color: #e0f2fe !important;
    font-weight: 600 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Text', sans-serif !important;
    font-variant-numeric: tabular-nums !important;
    letter-spacing: -0.01em !important;
    line-height: 1.35 !important;
    vertical-align: baseline !important;
    white-space: nowrap !important;
}

.fin-num-badge[data-type="range"] {
    border-color: rgba(125, 211, 252, 0.38) !important;
    font-weight: 650 !important;
}
.fin-num-badge[data-type="percent"] {
    font-variant-numeric: tabular-nums lining-nums !important;
}
.fin-num-badge[data-type="per-share"] {
    letter-spacing: 0 !important;
}

details.thought-accordion {
    margin: 6px 0 16px 0 !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(143, 138, 130, 0.2) !important;
    overflow: hidden !important;
    transition: all 0.2s ease !important;
}

details.thought-accordion:not(:has(.thought-body:not(:empty))) {
    display: none !important;
}

details.thought-accordion:hover {
    border-color: rgba(143, 138, 130, 0.35) !important;
    background: rgba(255, 255, 255, 0.035) !important;
}

details.thought-accordion.is-thinking {
    border: 1px solid rgba(143, 138, 130, 0.28) !important;
    background: rgba(255, 255, 255, 0.02) !important;
}

details.thought-accordion > summary {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 8px 14px !important;
    min-height: 38px !important;
    cursor: pointer !important;
    user-select: none !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 16.5px !important;
    font-weight: 650 !important;
    letter-spacing: -0.01em !important;
    color: #e4e4e7 !important;
    transition: color 0.2s, background-color 0.2s !important;
    list-style: none !important;
}

details.thought-accordion > summary::-webkit-details-marker {
    display: none !important;
}

details.thought-accordion > summary:hover {
    color: #f4f4f5 !important;
}

.thought-summary-inner {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.thought-chevron {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 15px !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #a1a1aa !important;
    transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), color 0.2s !important;
    transform: rotate(0deg);
}

details.thought-accordion[open] > summary .thought-chevron {
    transform: rotate(180deg) !important;
    color: #e4e4e7 !important;
}

.thought-label-text {
    font-size: 16.5px !important;
    font-weight: 650 !important;
    letter-spacing: -0.01em !important;
    color: #e4e4e7 !important;
}


@keyframes thinkingHighlightSweep {
    0% {
        background-position: -200% 0;
    }
    100% {
        background-position: 200% 0;
    }
}

details.thought-accordion.is-thinking .thought-label-text {
    background: linear-gradient(
        90deg,
        #a1a1aa 0%,
        #e4e4e7 30%,
        #ffffff 50%,
        #e4e4e7 70%,
        #a1a1aa 100%
    );
    background-size: 200% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: thinkingHighlightSweep 1.2s linear infinite;
    display: inline-block;
}

details.thought-accordion:not(.is-thinking) .thought-label-text {
    color: #e4e4e7 !important;
    background: none !important;
    -webkit-text-fill-color: #e4e4e7 !important;
    animation: none !important;
}

.thought-body {
    padding: 8px 18px 16px 18px !important;
    margin: 0 !important;
    border-left: none !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif !important;
    font-size: 15.5px !important;
    line-height: 1.62 !important;
    color: rgba(220, 215, 205, 0.9) !important;
    white-space: normal !important;
    word-break: break-word !important;
    max-height: 460px !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(143, 138, 130, 0.3) transparent !important;
}

.thought-body::-webkit-scrollbar {
    width: 6px !important;
}
.thought-body::-webkit-scrollbar-track {
    background: transparent !important;
}
.thought-body::-webkit-scrollbar-thumb {
    background: rgba(143, 138, 130, 0.3) !important;
    border-radius: 4px !important;
}

.thought-body p {
    margin: 0 0 8px 0 !important;
}

.thought-body ul, .thought-body ol {
    margin: 4px 0 8px 0 !important;
    padding-left: 18px !important;
}

.thought-body li {
    margin-bottom: 3px !important;
}


.rag-stage-indicator {
    display: inline-flex !important;
    align-items: center !important;
    padding: 6px 0 !important;
    margin: 2px 0 8px 0 !important;
    animation: ragStageFadeIn 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
}

.rag-stage-text {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    color: #b2b2b8 !important;
    background: linear-gradient(105deg,
        #a5a5ab 0%,
        #b2b2b8 37%,
        #d2d2d6 46%,
        #ffffff 50%,
        #d2d2d6 54%,
        #b2b2b8 63%,
        #a5a5ab 100%);
    background-size: 240% 100%;
    background-position: 100% 0;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: ragStatusGlowSweep 2.6s linear infinite !important;
    display: inline-flex !important;
    align-items: center !important;
}

@keyframes ragStatusGlowSweep {
    from { background-position: 100% 0; }
    to { background-position: 0% 0; }
}

@keyframes textBreathing {
    0%, 100% { opacity: 0.65; }
    50% { opacity: 1; }
}

@keyframes ragStageFadeIn {
    0% { opacity: 0; transform: translateY(2px); }
    100% { opacity: 1; transform: translateY(0); }
}


details.rag-sources-accordion {
    margin: 16px 0 10px 0 !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.015) !important;
    border: 1px solid rgba(143, 138, 130, 0.2) !important;
    overflow: hidden !important;
    transition: all 0.2s ease !important;
}

details.rag-sources-accordion:hover {
    border-color: rgba(143, 138, 130, 0.35) !important;
    background: rgba(255, 255, 255, 0.025) !important;
}

details.rag-sources-accordion > summary {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 9px !important;
    padding: 8px 14px !important;
    min-height: 36px !important;
    cursor: pointer !important;
    user-select: none !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    color: #94a3b8 !important;
    transition: color 0.2s !important;
    list-style: none !important;
}

details.rag-sources-accordion > summary::-webkit-details-marker {
    display: none !important;
}

details.rag-sources-accordion > summary:hover {
    color: #e2e8f0 !important;
}

.rag-sources-title-wrap {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.rag-sources-chevron {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 12px !important;
    font-size: 17px !important;
    font-family: monospace, sans-serif !important;
    font-weight: 700 !important;
    color: #8f8a82 !important;
    transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    transform: rotate(0deg) !important;
}

details.rag-sources-accordion[open] > summary .rag-sources-chevron {
    transform: rotate(90deg) !important;
}

.rag-sources-grid {
    padding: 6px 14px 14px 14px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    border-top: 1px solid rgba(143, 138, 130, 0.12) !important;
}


.rag-source-card {
    background: rgba(18, 18, 18, 0.6) !important;
    border: 1px solid rgba(143, 138, 130, 0.18) !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    transition: border-color 0.2s ease, background-color 0.2s ease !important;
}

.rag-source-card:hover {
    border-color: rgba(143, 138, 130, 0.35) !important;
    background: rgba(26, 26, 26, 0.75) !important;
}

.rag-source-header {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
}

.rag-source-filename {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
}

.rag-source-badge {
    font-size: 10px !important;
    font-weight: 600 !important;
    padding: 2px 7px !important;
    border-radius: 6px !important;
    letter-spacing: 0.03em !important;
    line-height: 1.3 !important;
}

.rag-source-badge-fy {
    background: rgba(56, 189, 248, 0.12) !important;
    color: #7dd3fc !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
}

.rag-source-badge-score {
    background: rgba(143, 138, 130, 0.12) !important;
    color: #d6d3cd !important;
    border: 1px solid rgba(143, 138, 130, 0.25) !important;
    font-family: monospace !important;
}

.rag-source-snippet {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 11.5px !important;
    line-height: 1.5 !important;
    color: rgba(203, 213, 225, 0.8) !important;
    font-style: italic !important;
    word-break: break-word !important;
}

@media (prefers-reduced-motion: reduce) {
    .rag-stage-text,
    .animated-dots .dot {
        animation: none !important;
        opacity: 1 !important;
        background: none !important;
        -webkit-text-fill-color: #e2e8f0 !important;
    }
    .rag-stage-indicator {
        animation: none !important;
    }
    .rag-trace-live, .rag-trace-live .rag-process-step, .rag-trace-live .rag-process-active,
    .rag-trace-live .rag-process-dot, .rag-trace-final, .post-generation-item {
        animation: none !important;
    }
    .rag-trace-collapse-after .rag-trace-content {
        animation: none !important;
    }
}

.agent-response-text,
.agent-response-text p,
.agent-response-text li,
.agent-response-text td,
.agent-response-text th,
.agent-response-text blockquote,
.agent-response-text strong,
.agent-response-text em,
.agent-response-text b,
.agent-response-text i {
    font-family: 'AG', Georgia, serif !important;
}

.agent-response-text {
    font-size: 18.5px !important;
    line-height: 1.7 !important;
    letter-spacing: 0.01em !important;
}

.agent-response-text p {
    margin: 0.45em 0 0.8em !important;
}
.agent-response-text h1,
.agent-response-text h2,
.agent-response-text h3 {
    font-family: 'AG', Georgia, serif !important;
    color: #f4f4f5 !important;
    line-height: 1.25 !important;
    letter-spacing: -0.015em !important;
    margin: 1.05em 0 0.38em !important;
}
.agent-response-text h1 { font-size: 1.35em !important; }
.agent-response-text h2 { font-size: 1.2em !important; }
.agent-response-text h3 { font-size: 1.08em !important; }
.agent-response-text ul,
.agent-response-text ol {
    margin: 0.35em 0 0.85em !important;
    padding-left: 1.45em !important;
}
.agent-response-text li {
    padding-left: 0.12em !important;
    margin: 0.18em 0 !important;
}

.agent-response-text strong,
.agent-response-text b {
    font-weight: 700 !important;
}

.agent-response-text em,
.agent-response-text i {
    font-style: italic !important;
}

.agent-response-text code,
.agent-response-text pre {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 14.5px !important;
}
</style>
<div class="agent-response-wrapper" style="
    padding: 32px 48px;
    background-image: url('data:image/svg+xml,%3csvg width=%22100%25%22 height=%22100%25%22 xmlns=%22http://www.w3.org/2000/svg%22%3e%3crect x=%220.75%22 y=%220.75%22 width=%22calc(100%25 - 1.5px)%22 height=%22calc(100%25 - 1.5px)%22 fill=%22none%22 rx=%2224%22 ry=%2224%22 stroke=%22%238f8a82%22 stroke-width=%221.5%22 stroke-dasharray=%224 8%22 /%3e%3c/svg%3e');
    border-radius: 24px;
    margin-bottom: 8px;
    margin-top: -32px;
">
    <div style="display: flex; gap: 16px; align-items: flex-start;">
        <div class="agent-blob"></div>
        <div style="display: flex; flex-direction: column; width: 100%;">
            <div class="ragscope-brand-anchor">RAGSCOPE</div>
            <div class="agent-response-text" style="font-family: 'AG', Georgia, serif; color: var(--st-text-color); font-size: 18.5px; line-height: 1.7; letter-spacing: 0.01em;">
"""


def render_agent_response_prefix(started_at_ms=None, worked_for_seconds=None):
    
    brand = '<div class="ragscope-brand-anchor">RAGSCOPE'
    if started_at_ms is not None:
        counter = (
            f'<span class="rag-working-counter" data-working-start="{int(started_at_ms)}">'
            'Working for 0s</span>'
        )
    elif worked_for_seconds is not None:
        seconds = max(0, int(round(float(worked_for_seconds))))
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            elapsed = f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            elapsed = f"{minutes}m {seconds}s"
        else:
            elapsed = f"{seconds}s"
        counter = f'<span class="rag-working-counter" data-working-final="true">Worked for {elapsed}</span>'
    else:
        counter = ""
    return AGENT_RESPONSE_PREFIX.replace(
        '<div class="ragscope-brand-anchor">RAGSCOPE</div>',
        brand + counter + '</div>',
    )

AGENT_RESPONSE_SUFFIX_STREAM = """</div>
</div>
</div>
</div>"""

DISABLE_SIDEBAR_STYLE = """
<style>
[data-testid="stSidebar"] {
    pointer-events: none !important;
    opacity: 0.6 !important;
    transition: opacity 0.3s ease !important;
}
</style>
"""

__all__ = [
    "GLOBAL_NAV_ARROWS_HTML",
    "USER_CARD_HTML",
    "USER_CARD_JS",
    "AGENT_RESPONSE_PREFIX",
    "render_agent_response_prefix",
    "AGENT_RESPONSE_SUFFIX_STREAM",
    "DISABLE_SIDEBAR_STYLE",
]
