document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const hfStatus = document.getElementById('hf-status');
    const hfText = document.getElementById('hf-text');
    const groqStatus = document.getElementById('groq-status');
    const groqText = document.getElementById('groq-text');
    const setupWarning = document.getElementById('setup-warning');

    const promptInput = document.getElementById('prompt-input');
    const charCounter = document.getElementById('char-counter');
    const generateBtn = document.getElementById('generate-btn');
    const chipBtns = document.querySelectorAll('.chip-btn');

    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    const errorSection = document.getElementById('error-section');

    const generatedImage = document.getElementById('generated-image');
    const resultOriginalPrompt = document.getElementById('result-original-prompt');
    const resultEnhancedPrompt = document.getElementById('result-enhanced-prompt');
    const copyEnhancedBtn = document.getElementById('copy-enhanced-btn');

    const downloadBtn = document.getElementById('download-btn');
    const generateAgainBtn = document.getElementById('generate-again-btn');
    const fullscreenBtn = document.getElementById('fullscreen-btn');

    const errorTitle = document.getElementById('error-title');
    const errorMessage = document.getElementById('error-message');
    const errorDismissBtn = document.getElementById('error-dismiss-btn');

    const historyGrid = document.getElementById('history-grid');
    const historyEmpty = document.getElementById('history-empty');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modal-img');
    const modalClose = document.getElementById('modal-close');

    let currentImageData = null;

    // --- 1. Check API Health & Credentials Status ---
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();

            // Hugging Face Status
            if (data.hf_configured) {
                hfStatus.querySelector('.dot').className = 'dot green';
                hfText.textContent = 'Ready';
                setupWarning.classList.add('hidden');
            } else {
                hfStatus.querySelector('.dot').className = 'dot red';
                hfText.textContent = 'Missing Token';
                setupWarning.classList.remove('hidden');
            }

            // Groq LLM Status
            if (data.groq_configured) {
                groqStatus.querySelector('.dot').className = 'dot green';
                groqText.textContent = 'Active (' + data.groq_model.split('-')[0] + ')';
            } else {
                groqStatus.querySelector('.dot').className = 'dot yellow';
                groqText.textContent = 'Fallback Mode';
            }
        } catch (err) {
            console.warn('Health check failed:', err);
            hfText.textContent = 'Offline';
            groqText.textContent = 'Offline';
        }
    }

    checkHealth();

    // --- 2. Input Handling & Character Counter ---
    promptInput.addEventListener('input', () => {
        const count = promptInput.value.length;
        charCounter.textContent = `${count} / 1000`;
    });

    chipBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const prompt = btn.getAttribute('data-prompt');
            promptInput.value = prompt;
            promptInput.dispatchEvent(new Event('input'));
            promptInput.focus();
        });
    });

    // --- 3. Stepper Progress Animation ---
    function resetSteps() {
        for (let i = 1; i <= 4; i++) {
            const step = document.getElementById(`step-${i}`);
            step.className = 'step-item';
        }
    }

    function animateSteps() {
        resetSteps();
        
        // Step 1: Intent Analysis
        document.getElementById('step-1').className = 'step-item active';

        // Step 2: Groq Prompt Optimization
        setTimeout(() => {
            document.getElementById('step-1').className = 'step-item completed';
            document.getElementById('step-2').className = 'step-item active';
        }, 700);

        // Step 3: FLUX.1 Schnell Generation
        setTimeout(() => {
            document.getElementById('step-2').className = 'step-item completed';
            document.getElementById('step-3').className = 'step-item active';
        }, 1600);

        // Step 4: Finalizing
        setTimeout(() => {
            document.getElementById('step-3').className = 'step-item completed';
            document.getElementById('step-4').className = 'step-item active';
        }, 3200);
    }

    // --- 4. Generate Image Request ---
    async function handleGenerate() {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            showError('Empty Prompt', 'Please enter a description for the image you want to create.');
            return;
        }

        // UI Loading State
        setLoadingState(true);
        hideAllSections();
        loadingSection.classList.remove('hidden');
        animateSteps();

        try {
            const response = await fetch('/api/generate-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to generate image');
            }

            // Complete steps animation
            document.getElementById('step-4').className = 'step-item completed';

            setTimeout(() => {
                displayResult(data);
                saveToHistory(data);
                setLoadingState(false);
            }, 500);

        } catch (err) {
            console.error('Generation Error:', err);
            setLoadingState(false);
            showError('Generation Error', err.message || 'An error occurred while generating your image.');
        }
    }

    generateBtn.addEventListener('click', handleGenerate);

    // Ctrl+Enter keyboard shortcut
    promptInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            handleGenerate();
        }
    });

    // --- 5. UI Helpers ---
    function setLoadingState(isLoading) {
        generateBtn.disabled = isLoading;
        const text = generateBtn.querySelector('.btn-text');
        const spinner = generateBtn.querySelector('.btn-spinner');
        if (isLoading) {
            text.classList.add('hidden');
            spinner.classList.remove('hidden');
        } else {
            text.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }

    function hideAllSections() {
        loadingSection.classList.add('hidden');
        resultSection.classList.add('hidden');
        errorSection.classList.add('hidden');
    }

    function displayResult(data) {
        hideAllSections();
        resultSection.classList.remove('hidden');

        currentImageData = data;
        generatedImage.src = data.image;
        resultOriginalPrompt.textContent = data.original_prompt;
        resultEnhancedPrompt.textContent = data.enhanced_prompt;

        // Scroll smoothly to result
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showError(title, msg) {
        hideAllSections();
        errorSection.classList.remove('hidden');
        errorTitle.textContent = title;
        errorMessage.textContent = msg;
    }

    errorDismissBtn.addEventListener('click', () => {
        errorSection.classList.add('hidden');
        promptInput.focus();
    });

    generateAgainBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        promptInput.focus();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // --- 6. Image Download ---
    downloadBtn.addEventListener('click', () => {
        if (!currentImageData || !currentImageData.image) return;

        const link = document.createElement('a');
        link.href = currentImageData.image;
        
        // Generate filename from prompt slug
        const promptSlug = (currentImageData.original_prompt || 'ai-image')
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .slice(0, 30);
            
        link.download = `flux-${promptSlug}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Copy Enhanced Prompt
    copyEnhancedBtn.addEventListener('click', () => {
        const text = resultEnhancedPrompt.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalHTML = copyEnhancedBtn.innerHTML;
            copyEnhancedBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
            setTimeout(() => {
                copyEnhancedBtn.innerHTML = originalHTML;
            }, 2000);
        });
    });

    // Fullscreen View
    fullscreenBtn.addEventListener('click', () => {
        if (generatedImage.src) {
            modalImg.src = generatedImage.src;
            modal.classList.remove('hidden');
        }
    });

    modalClose.addEventListener('click', () => {
        modal.classList.add('hidden');
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });

    // --- 7. LocalStorage Generation History ---
    const HISTORY_KEY = 'flux_agent_history';

    function loadHistory() {
        const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        historyGrid.innerHTML = '';

        if (history.length === 0) {
            historyEmpty.classList.remove('hidden');
            return;
        }

        historyEmpty.classList.add('hidden');

        history.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'history-card';
            
            const timeAgo = formatTime(item.timestamp);

            card.innerHTML = `
                <img class="history-thumb" src="${item.image}" alt="Thumbnail" loading="lazy">
                <div class="history-info">
                    <p class="history-prompt">${escapeHtml(item.original_prompt)}</p>
                    <span class="history-time">${timeAgo}</span>
                </div>
            `;

            card.addEventListener('click', () => {
                displayResult(item);
            });

            historyGrid.appendChild(card);
        });
    }

    function saveToHistory(item) {
        let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        
        // Add new item to beginning
        history.unshift({
            original_prompt: item.original_prompt,
            enhanced_prompt: item.enhanced_prompt,
            image: item.image,
            timestamp: item.timestamp || new Date().toISOString()
        });

        // Limit to max 20 history items to save localStorage memory
        if (history.length > 20) {
            history = history.slice(0, 20);
        }

        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
            loadHistory();
        } catch (e) {
            console.warn('LocalStorage full, quota exceeded:', e);
        }
    }

    clearHistoryBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear your generation history?')) {
            localStorage.removeItem(HISTORY_KEY);
            loadHistory();
        }
    });

    function formatTime(isoString) {
        if (!isoString) return 'Just now';
        const date = new Date(isoString);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Initial load of history
    loadHistory();
});
