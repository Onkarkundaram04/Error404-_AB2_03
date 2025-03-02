document.addEventListener('DOMContentLoaded', function () {
    // Navigation handling
    const links = document.querySelectorAll('a.menu-item');
    links.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            window.location.href = href;
        });
    });

    // Active page highlighting
    const currentPage = window.location.pathname.split('/').pop();
    const activeLink = document.querySelector(`a[href="${currentPage}"]`);
    if (activeLink) {
        document.querySelectorAll('.menu-item').forEach(item => {
            item.classList.remove('active');
        });
        activeLink.classList.add('active');
    }

    // Upload handling
    window.uploadPDF = async function(files) {
        if (!files || files.length === 0) return;

        const uploadLabel = document.getElementById('uploadLabel');
        const uploadText = document.getElementById('uploadText');
        const fileInput = document.getElementById('file-upload');
        const statusDiv = document.getElementById('uploadStatus');

        // Reset previous state
        statusDiv.style.display = 'none';
        uploadLabel.style.display = 'block';
        uploadLabel.style.backgroundColor = '#f0f0f0';
        uploadText.textContent = 'Processing PDF...';
        fileInput.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', files[0]);

            const response = await fetch('http://localhost:5000/upload-pdf', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error');
            }

            // Update UI
            uploadLabel.style.display = 'none';
            statusDiv.style.display = 'block';
            statusDiv.style.color = '#4CAF50';
            statusDiv.textContent = '✓ Analysis Complete';
            
            // Update patient details
            document.getElementById('patientName').textContent = data.patientName;
            document.getElementById('patientAge').textContent = `${data.age} years`;
            document.getElementById('patientHistory').textContent = data.medicalHistory;

            // Update recommendations
            const recommendationsDiv = document.getElementById('recommendations');
            recommendationsDiv.innerHTML = '';
            data.recommendations.forEach((rec, index) => {
                const card = document.createElement('div');
                card.className = 'recommendation-card';
                card.innerHTML = `
                    <h3>Recommendation ${index + 1}</h3>
                    <p>${rec}</p>
                `;
                recommendationsDiv.appendChild(card);
            });

        } catch (error) {
            statusDiv.style.display = 'block';
            statusDiv.style.color = '#f44336';
            statusDiv.textContent = '✗ Upload Failed';
            uploadText.textContent = 'Try Again';
            console.error('Upload error:', error);
            alert(`Error: ${error.message}`);
        } finally {
            fileInput.disabled = false;
            uploadLabel.style.backgroundColor = '';
        }
    };
});