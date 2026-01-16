document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const searchBtn = document.getElementById('search-btn');
    const previewImg = document.getElementById('preview-img');
    const resultsArea = document.getElementById('results-area');
    const loading = document.getElementById('loading');
    const limitInput = document.getElementById('limit-input');

    let currentFile = null;

    // 拖拽事件
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    // 可见按钮也触发文件选择（更兼容）
    const chooseBtn = document.getElementById('choose-btn');
    if (chooseBtn) chooseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (!file.type.startsWith('image/')) {
                alert("请上传图片文件");
                return;
            }
            currentFile = file;
            showPreview(file);
            searchBtn.disabled = false;
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            previewImg.src = reader.result;
            previewImg.hidden = false;
            // 隐藏上传提示内容
            dropZone.querySelector('.upload-content').style.opacity = '0';
        }
    }

    searchBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI 状态更新
        resultsArea.innerHTML = '';
        loading.hidden = false;
        searchBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('limit', limitInput.value);

        try {
            const response = await fetch('/search', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loading.hidden = true;
            searchBtn.disabled = false;

            if (response.ok) {
                renderResults(data.results);
            } else {
                resultsArea.innerHTML = `<p class="error">出错啦: ${data.error || '未知错误'}</p>`;
            }

        } catch (error) {
            console.error('Error:', error);
            loading.hidden = true;
            searchBtn.disabled = false;
            resultsArea.innerHTML = `<p class="error">网络错误，请稍后再试。</p>`;
        }
    });

    function renderResults(results) {
        if (!results || results.length === 0) {
            resultsArea.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #666;">未找到相似图片</p>';
            return;
        }

        results.forEach((item, index) => {
            // 动画延迟
            const delay = index * 0.1;
            
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animation = `fadeIn 0.5s ease ${delay}s backwards`;

            // 修正图片路径：如果后端返回的是相对路径，这里可能需要调整
            // 假设后端返回 "gallery/images/xxxx.jpg"
            // 我们通过 /gallery/images/xxxx.jpg 访问
            // 规范化 image_path：替换反斜杠并移除多余前缀
            let imgUrl = item.image_path || '';
            imgUrl = imgUrl.replace(/\\/g, '/');
            if (imgUrl.startsWith('/')) imgUrl = imgUrl.slice(1);
            if (imgUrl.startsWith('gallery/')) imgUrl = imgUrl.slice('gallery/'.length);

            const displayScore = (item.score * 100).toFixed(1) + '%';
            
            card.innerHTML = `
                <a href="${item.image_url || ('/gallery/' + imgUrl)}" target="_blank" title="查看原图">
                    <img src="/gallery/${imgUrl}" class="result-img" alt="Similar Image" loading="lazy">
                </a>
                <div class="result-info">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="score-badge">相似度: ${displayScore}</span>
                        <span style="font-size: 0.8em; color: #999;">#${index + 1}</span>
                    </div>
                </div>
            `;
            
            resultsArea.appendChild(card);
        });
    }
});
