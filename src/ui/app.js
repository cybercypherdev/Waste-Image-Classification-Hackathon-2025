// DOM Elements
console.log("app.js has loaded!");

const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const closeDrawerBtn = document.getElementById('closeDrawer');
const sidebar = document.getElementById('sidebar');
const drawerOverlay = document.getElementById('drawerOverlay');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInputBtn = document.getElementById('fileInputBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultSection = document.getElementById('resultSection');
const previewImage = document.getElementById('previewImage');
const previewContainer = document.getElementById('previewContainer');
const errorMessage = document.getElementById('errorMessage');
const cameraFeed = document.getElementById('cameraFeed');
const startCameraBtn = document.getElementById('startCamera');
const captureBtn = document.getElementById('captureBtn');
const resetBtn = document.getElementById('resetBtn');
const recyclableCount = document.getElementById('recyclableCount');
const organicCount = document.getElementById('organicCount');
const finalPrediction = document.getElementById('finalPrediction');
const confidenceValue = document.getElementById('confidenceValue');
const modelConfidence = document.getElementById('modelConfidence');
const processingTime = document.getElementById('processingTime');
const modelAccuracy = document.getElementById('modelAccuracy');
const homeSection = document.getElementById('homeSection');
const interfaceContainer = document.querySelector('.interface-container');
const carouselSlides = document.getElementById('carouselSlides');
const carouselDots = document.getElementById('carouselDots');
const prevSlideBtn = document.getElementById('prevSlide');
const nextSlideBtn = document.getElementById('nextSlide');
const uploadSection = document.getElementById('uploadSection');
const cameraSection = document.getElementById('cameraSection');
const statsSection = document.getElementById('statsSection');
const aboutSection = document.getElementById('aboutSection');
const readmeContent = document.getElementById('readmeContent');

// State
let isCameraActive = false;
let currentStream = null;
let classificationCounts = {
    recyclable: 0,
    organic: 0
};

// Carousel State
let currentSlide = 0;
let slides = [];
let autoPlayInterval;

// API Configuration
const API_URL = 'http://localhost:8000';

// Fallback carousel images
const fallbackImages = [
    { url: '/images/african-american-woman-recycling-better-environment.jpg' },
    { url: '/images/black-bags-trash-garbage-bin-daytime.jpg' },
    { url: '/images/close-up-hand-holding-plastic-bottle.jpg' },
    { url: '/images/leftover-food-waste-recycle-bin.jpg' },
    { url: '/images/many-kinds-garbage-were-collected-dark-floor.jpg' },
    { url: '/images/recycle-concept-with-box-rubbish.jpg' },
    { url: '/images/recycling-concept-flat-lay.jpg' },
    { url: '/images/recycling-with-plastic-bottles-cans-container.jpg' },
    { url: '/images/sustainable-development-goals-still-life.jpg' },
    { url: '/images/woman-recycling-better-environment.jpg' }
];




// Initialize particles
function initParticles() {
    const container = document.querySelector('.particles-container');
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDuration = (Math.random() * 3 + 2) + 's';
        container.appendChild(particle);
    }
}

// Initialize
function init() {
    initParticles();
    loadCarouselImages();
    showSection('home');
    setupEventListeners();
}

// Event Listeners Setup
function setupEventListeners() {
    console.log("Setting up event listeners...");

    if (!mobileMenuBtn || !closeDrawerBtn || !drawerOverlay) {
        console.error("One or more elements are missing!");
        return;
    }

    mobileMenuBtn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation(); // Prevents accidental closing
        console.log("Mobile menu button clicked - FORCING drawer to stay open");
    
        openDrawer();
    
        setTimeout(() => {
            console.log("Drawer should now be visible.");
        }, 500); // Delay to confirm it's opening
    });
    

    closeDrawerBtn.addEventListener('click', () => {
        console.log("Close drawer button clicked");
        closeDrawer();
    });

    drawerOverlay.addEventListener('click', (event) => {
        event.stopPropagation(); // Prevents it from triggering other clicks
        console.log("Overlay clicked, but will not close immediately");
        
        setTimeout(() => {
            closeDrawer();
        }, 300); // Delay closing slightly to ensure menu stays open briefly
    });
    
    // Mobile menu
    mobileMenuBtn.addEventListener('click', openDrawer);
    closeDrawerBtn.addEventListener('click', closeDrawer);
    drawerOverlay.addEventListener('click', closeDrawer);

    // File upload
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) {
            console.log('File dropped:', file.name);
            handleFile(file);
        }
    });

    fileInputBtn.addEventListener('click', () => {
        console.log('File input button clicked');
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file.name);
            handleFile(file);
        }
    });

    // Camera controls
    startCameraBtn.addEventListener('click', () => {
        console.log('Camera button clicked');
        toggleCamera();
    });

    captureBtn.addEventListener('click', () => {
        console.log('Capture button clicked');
        captureAndClassify();
    });

    // Reset button
    resetBtn.addEventListener('click', resetUI);

    // Carousel controls
    prevSlideBtn.addEventListener('click', () => {
        prevSlide();
        stopAutoPlay();
        startAutoPlay();
    });

    nextSlideBtn.addEventListener('click', () => {
        nextSlide();
        stopAutoPlay();
        startAutoPlay();
    });

    // Navigation
    document.querySelectorAll('.nav-links li').forEach(item => {
        item.addEventListener('click', () => {
            const section = item.dataset.section;
            showSection(section);
            
            if (window.innerWidth <= 768) {
                closeDrawer();
            }
        });
    });
}

// Camera Functions
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraFeed.srcObject = stream;
        currentStream = stream;
        isCameraActive = true;
        startCameraBtn.textContent = 'Stop Camera';
        captureBtn.disabled = false;
    } catch (error) {
        console.error('Error accessing camera:', error);
        showError('Could not access camera. Please check permissions.');
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
        isCameraActive = false;
        startCameraBtn.textContent = 'Start Camera';
        captureBtn.disabled = true;
    }
}

function toggleCamera() {
    if (isCameraActive) {
        stopCamera();
    } else {
        startCamera();
    }
}

function captureAndClassify() {
    if (!isCameraActive) {
        showError('Please start the camera first.');
        return;
    }

    console.log('Capturing image from camera');
    
    const canvas = document.createElement('canvas');
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    canvas.getContext('2d').drawImage(cameraFeed, 0, 0);

    canvas.toBlob((blob) => {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        handleFile(file, 'camera');
    }, 'image/jpeg');
}

// File Handling
function handleFile(file, section = 'upload') {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file.');
        return;
    }

    console.log('Handling file:', file.name, 'Type:', file.type);

    const reader = new FileReader();
    reader.onload = (e) => {
        if (section === 'upload') {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            dropZone.style.display = 'none';
        } else if (section === 'camera') {
            const cameraPreview = document.createElement('img');
            cameraPreview.src = e.target.result;
            cameraPreview.style.width = '100%';
            cameraPreview.style.borderRadius = '8px';
            cameraPreview.style.marginBottom = '1rem';
            
            const existingPreview = cameraSection.querySelector('img');
            if (existingPreview) {
                existingPreview.remove();
            }
            cameraSection.insertBefore(cameraPreview, cameraSection.querySelector('.camera-controls'));
        }
        uploadImage(file, section);
    };
    reader.readAsDataURL(file);
}

// API Integration
async function uploadImage(file, section = 'upload') {
    try {
        showLoading();
        hideError();

        const formData = new FormData();
        formData.append('file', file);

        console.log('Uploading image to:', `${API_URL}/predict`);

        const startTime = performance.now();
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Received response:', data);
        
        const endTime = performance.now();

        // Update processing time
        processingTime.textContent = `${(endTime - startTime).toFixed(2)}ms`;

        // Display results in appropriate section
        if (section === 'upload') {
            displayResults(data, resultSection);
        } else if (section === 'camera') {
            displayResults(data, document.getElementById('cameraResultSection'));
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to process image. Please check if the server is running at ' + API_URL);
    } finally {
        hideLoading();
    }
}

// UI Updates
function displayResults(data, resultSection) {
    resultSection.style.display = 'block';
    
    // Update confidence bars
    const recyclableBar = resultSection.querySelector('.bar-fill.recyclable');
    const organicBar = resultSection.querySelector('.bar-fill.organic');
    const recyclableValue = resultSection.querySelector('.result-card:first-child .confidence-value');
    const organicValue = resultSection.querySelector('.result-card:last-child .confidence-value');

    recyclableBar.style.width = `${data.predictions.Recyclable * 100}%`;
    organicBar.style.width = `${data.predictions.Organic * 100}%`;
    recyclableValue.textContent = `${(data.predictions.Recyclable * 100).toFixed(1)}%`;
    organicValue.textContent = `${(data.predictions.Organic * 100).toFixed(1)}%`;

    // Update final prediction
    const finalPrediction = resultSection.querySelector('.final-prediction span');
    const confidenceValue = resultSection.querySelector('.confidence span');
    finalPrediction.textContent = data.prediction;
    confidenceValue.textContent = `${(data.probability * 100).toFixed(1)}%`;

    // Update counts
    if (data.prediction === 'Recyclable') {
        classificationCounts.recyclable++;
        recyclableCount.textContent = classificationCounts.recyclable;
    } else {
        classificationCounts.organic++;
        organicCount.textContent = classificationCounts.organic;
    }

    // Update model confidence
    modelConfidence.textContent = `${(data.probability * 100).toFixed(1)}%`;
}

function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showError(message) {
    errorMessage.querySelector('span').textContent = message;
    errorMessage.style.display = 'flex';
}

function hideError() {
    errorMessage.style.display = 'none';
}

function resetUI() {
    previewContainer.style.display = 'none';
    dropZone.style.display = 'block';
    resultSection.style.display = 'none';
    hideError();
}

// Drawer Functions
function openDrawer() {
    sidebar.classList.add('active');
    drawerOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Ensure overlay is clickable when needed
    drawerOverlay.style.display = 'block';
}

function closeDrawer() {
    sidebar.classList.remove('active');
    drawerOverlay.classList.remove('active');
    document.body.style.overflow = '';

    // Ensure overlay is removed from blocking interactions
    setTimeout(() => {
        drawerOverlay.style.display = 'none';
    }, 300); 
}


// Carousel Functions
function goToSlide(index) {
    currentSlide = index;
    updateCarousel();
}

function nextSlide() {
    currentSlide = (currentSlide + 1) % slides.length;
    updateCarousel();
}

function prevSlide() {
    currentSlide = (currentSlide - 1 + slides.length) % slides.length;
    updateCarousel();
}

function updateCarousel() {
    carouselSlides.style.transform = `translateX(-${currentSlide * 100}%)`;
    document.querySelectorAll('.carousel-dot').forEach((dot, index) => {
        dot.classList.toggle('active', index === currentSlide);
    });
}

function startAutoPlay() {
    autoPlayInterval = setInterval(nextSlide, 5000);
}

function stopAutoPlay() {
    clearInterval(autoPlayInterval);
}

// Section Navigation
function showSection(section) {
    // Hide all sections
    homeSection.style.display = 'none';
    interfaceContainer.style.display = 'none';
    uploadSection.style.display = 'none';
    cameraSection.style.display = 'none';
    statsSection.style.display = 'none';
    aboutSection.style.display = 'none';

    // Show appropriate section
    switch (section) {
        case 'home':
            homeSection.style.display = 'flex';
            break;
        case 'upload':
            interfaceContainer.style.display = 'grid';
            uploadSection.style.display = 'block';
            if (isCameraActive) stopCamera();
            break;
        case 'camera':
            interfaceContainer.style.display = 'grid';
            cameraSection.style.display = 'block';
            startCamera();
            break;
        case 'stats':
            interfaceContainer.style.display = 'grid';
            statsSection.style.display = 'block';
            updateStats();
            break;
        case 'about':
            interfaceContainer.style.display = 'grid';
            aboutSection.style.display = 'block';
            loadReadme();
            break;
    }
    
    // Update active state in navigation
    document.querySelectorAll('.nav-links li').forEach(li => {
        li.classList.toggle('active', li.dataset.section === section);
    });
}

// Stats Functions
function updateStats() {
    // Update total counts
    document.getElementById('totalRecyclable').textContent = classificationCounts.recyclable;
    document.getElementById('totalOrganic').textContent = classificationCounts.organic;
    document.getElementById('totalClassifications').textContent = 
        classificationCounts.recyclable + classificationCounts.organic;

    // Update chart if it exists
    if (window.classificationChart) {
        window.classificationChart.destroy();
    }

    const ctx = document.getElementById('classificationChart').getContext('2d');
    window.classificationChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Recyclable', 'Organic'],
            datasets: [{
                data: [classificationCounts.recyclable, classificationCounts.organic],
                backgroundColor: ['#00ff9d', '#00b8ff'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#ffffff'
                    }
                }
            }
        }
    });
}

// README Loading
async function loadReadme() {
    try {
        const response = await fetch(`Hackerthon\README.md`);
        const readme = await response.json();
        
        // Convert markdown to HTML
        const html = marked.parse(readme.content);
        readmeContent.innerHTML = html;
    } catch (error) {
        console.error('Error loading README:', error);
        readmeContent.innerHTML = '<p>Error loading README content. Please try again later.</p>';
    }
}

// Load Carousel Images
async function loadCarouselImages() {
    try {
        // Try to fetch from API first
        const response = await fetch(`${API_URL}/images`);
        const images = await response.json();
        createCarouselSlides(images);

    } catch (error) {
        console.log('Using fallback carousel images');
        createCarouselSlides(fallbackImages);
    }
}

function createCarouselSlides(images) {
    // Clear existing slides
    carouselSlides.innerHTML = '';
    carouselDots.innerHTML = '';
    slides = [];

    // Create slides
    images.forEach((image, index) => {
        const slide = document.createElement('div');
        slide.className = 'carousel-slide';
        slide.innerHTML = `<img src="${image.url}" alt="Slide ${index + 1}">`;
        carouselSlides.appendChild(slide);
        
        // Create dot
        const dot = document.createElement('div');
        dot.className = 'carousel-dot' + (index === 0 ? ' active' : '');
        dot.addEventListener('click', () => goToSlide(index));
        carouselDots.appendChild(dot);
        
        slides.push(slide);
    });
    
    // Start autoplay
    startAutoPlay();
}

// Initialize the application
init(); 