// Enhanced Modern AI Invoice Generator - JavaScript

let selectedFile = null;
let ollamaAvailable = false;

// DOM Elements
const attachBtn = document.getElementById('attachBtn');
const imageInput = document.getElementById('imageInput');
const uploadForm = document.getElementById('uploadForm');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingSubtext = document.getElementById('loadingSubtext');
const inputPlaceholder = document.getElementById('inputPlaceholder');
const fileInfo = document.getElementById('fileInfo');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const fileRemove = document.getElementById('fileRemove');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const ollamaStatus = document.getElementById('ollamaStatus');
const ollamaOption = document.getElementById('ollama-option');

// Check Ollama availability on load
async function checkOllamaAvailability() {
    try {
        const response = await fetch('/api/check-ollama');
        const data = await response.json();
        ollamaAvailable = data.available;
        
        if (ollamaAvailable) {
            ollamaStatus.classList.add('available');
            ollamaStatus.querySelector('.status-text').textContent = `${data.model} Ready`;
        } else {
            ollamaStatus.classList.add('unavailable');
            ollamaStatus.querySelector('.status-text').textContent = 'Not Available';
            ollamaOption.style.opacity = '0.5';
            ollamaOption.style.cursor = 'not-allowed';
            document.getElementById('method-ollama').disabled = true;
        }
    } catch (error) {
        console.error('Failed to check Ollama:', error);
        ollamaStatus.classList.add('unavailable');
        ollamaStatus.querySelector('.status-text').textContent = 'Check Failed';
    }
}

// Initialize
checkOllamaAvailability();

// Event Listeners
attachBtn.addEventListener('click', () => {
    imageInput.click();
});

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        
        // Show file info
        inputPlaceholder.style.display = 'none';
        fileInfo.style.display = 'flex';
        fileNameDisplay.textContent = file.name;
        sendBtn.disabled = false;
        
        // Animate button
        sendBtn.style.animation = 'pulse 0.5s ease';
        setTimeout(() => {
            sendBtn.style.animation = '';
        }, 500);
    }
});

fileRemove.addEventListener('click', () => {
    resetFileInput();
});

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
        showError('Veuillez sélectionner une image.');
        return;
    }
    
    // Get selected method
    const method = document.querySelector('input[name="method"]:checked').value;
    
    // Disable form
    sendBtn.disabled = true;
    sendBtn.classList.add('processing');
    attachBtn.disabled = true;
    
    // Show user message with image
    await showUserMessage(selectedFile);
    
    // Show loading
    showLoading(true, method);
    
    // Show progress
    showProgress(true);
    updateProgress(20, 'Uploading image...');
    
    // Create FormData
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('method', method);
    
    try {
        updateProgress(40, 'Extracting text with OCR...');
        
        const response = await fetch('/api/process-image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Erreur serveur');
        }
        
        updateProgress(80, 'Generating invoice...');
        
        // Delay for smooth UX
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateProgress(100, 'Complete!');
        
        // Hide loading
        showLoading(false);
        showProgress(false);
        
        // Show bot response with results
        showBotResponse(data);
        
        // Reset form
        resetForm();
        
    } catch (error) {
        showLoading(false);
        showProgress(false);
        showBotError(error.message);
        resetForm();
    }
});

// Helper Functions

function showProgress(show) {
    progressContainer.style.display = show ? 'block' : 'none';
    if (!show) {
        progressFill.style.width = '0%';
    }
}

function updateProgress(percent, text) {
    progressFill.style.width = percent + '%';
    progressText.textContent = text;
}

function showUserMessage(file) {
    return new Promise((resolve) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message fade-in';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar user-avatar';
        avatar.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="8" r="4" fill="white"/>
                <path d="M4 20C4 16.6863 6.68629 14 10 14H14C17.3137 14 20 16.6863 20 20V21H4V20Z" fill="white"/>
            </svg>
        `;
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        // Read image as data URL
        const reader = new FileReader();
        reader.onload = (e) => {
            content.innerHTML = `
                <div class="message-header">
                    <strong>Vous</strong>
                    <span class="timestamp">now</span>
                </div>
                <p><strong>Voici mon image :</strong></p>
                <div class="image-preview">
                    <img src="${e.target.result}" alt="Uploaded image">
                </div>
            `;
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            chatContainer.appendChild(messageDiv);
            scrollToBottom();
            resolve();
        };
        reader.readAsDataURL(file);
    });
}

function showBotResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message fade-in';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bot-avatar';
    avatar.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" fill="white"/>
            <path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="#667eea" stroke-width="2" stroke-linecap="round"/>
            <circle cx="9" cy="9" r="1.5" fill="#667eea"/>
            <circle cx="15" cy="9" r="1.5" fill="#667eea"/>
        </svg>
    `;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    // Method badge
    const methodBadge = data.method_used === 'ollama' ? 
        '<span class="badge badge-smart">AI</span>' : 
        '<span class="badge badge-fast">Regex</span>';
    
    // Build response HTML
    let html = `
        <div class="message-header">
            <strong>🤖 AI Assistant ${methodBadge}</strong>
            <span class="timestamp">now</span>
        </div>
        <p><strong>✅ Traitement terminé avec succès !</strong></p>
    `;
    
    // Show extracted text
    if (data.extracted_text) {
        html += `
            <p><strong>📄 Texte extrait :</strong></p>
            <div class="extracted-text">${escapeHtml(data.extracted_text.substring(0, 500))}${data.extracted_text.length > 500 ? '...' : ''}</div>
        `;
    }
    
    // Show invoice data
    if (data.invoice_data) {
        const invoice = data.invoice_data;
        html += `
            <div class="invoice-data">
                <h4>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>
                    </svg>
                    Données de la facture
                </h4>
                <table>
                    <tr>
                        <td>N° Facture:</td>
                        <td>${escapeHtml(invoice.invoice_number)}</td>
                    </tr>
                    <tr>
                        <td>Date:</td>
                        <td>${escapeHtml(invoice.date)}</td>
                    </tr>
                    <tr>
                        <td>Client:</td>
                        <td>${escapeHtml(invoice.client_name)}</td>
                    </tr>
                    ${invoice.client_address ? `<tr>
                        <td>Adresse:</td>
                        <td>${escapeHtml(invoice.client_address)}</td>
                    </tr>` : ''}
                    <tr>
                        <td>Articles:</td>
                        <td>${invoice.items.length} article(s)</td>
                    </tr>
                    <tr>
                        <td>Total:</td>
                        <td><strong>${invoice.total.toFixed(2)} €</strong></td>
                    </tr>
                </table>
            </div>
        `;
    }
    
    // Download button
    if (data.pdf_filename) {
        html += `
            <button class="download-btn" onclick="downloadPDF('${data.pdf_filename}')">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
                </svg>
                Télécharger la facture PDF
            </button>
        `;
    }
    
    content.innerHTML = html;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function showBotError(errorMessage) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message fade-in';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bot-avatar';
    avatar.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" fill="white"/>
            <path d="M12 8v4M12 16h.01" stroke="#f56565" stroke-width="2" stroke-linecap="round"/>
        </svg>
    `;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `
        <div class="message-header">
            <strong>🤖 AI Assistant</strong>
            <span class="timestamp">now</span>
        </div>
        <p><strong>❌ Erreur</strong></p>
        <div class="error-message">
            ${escapeHtml(errorMessage)}
        </div>
        <p>Veuillez réessayer avec une autre image ou vérifier que l'image contient du texte lisible.</p>
    `;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function showError(message) {
    alert(message);
}

function showLoading(show, method = 'regex') {
    if (show) {
        loadingOverlay.classList.add('active');
        const methodText = method === 'ollama' ? 'AI (Llama 3.2:1b)' : 'Regex';
        loadingSubtext.textContent = `Using ${methodText} method`;
    } else {
        loadingOverlay.classList.remove('active');
    }
}

function scrollToBottom() {
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
}

function resetForm() {
    resetFileInput();
    attachBtn.disabled = false;
    sendBtn.classList.remove('processing');
}

function resetFileInput() {
    selectedFile = null;
    imageInput.value = '';
    fileInfo.style.display = 'none';
    inputPlaceholder.style.display = 'flex';
    sendBtn.disabled = true;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Download PDF function
window.downloadPDF = function(filename) {
    window.location.href = `/api/download/${filename}`;
};

// Auto-scroll to bottom on load
window.addEventListener('load', () => {
    scrollToBottom();
});

// Add ripple effect to buttons
document.querySelectorAll('.attach-btn, .send-btn').forEach(button => {
    button.addEventListener('click', function(e) {
        const ripple = this.querySelector('.btn-ripple');
        if (ripple) {
            ripple.style.left = e.offsetX + 'px';
            ripple.style.top = e.offsetY + 'px';
            ripple.style.animation = 'none';
            setTimeout(() => {
                ripple.style.animation = 'ripple 0.6s';
            }, 10);
        }
    });
});
