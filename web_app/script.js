document.addEventListener('DOMContentLoaded', () => {
    // --- Tab Switching Logic ---
    const tabs = document.querySelectorAll('.tab');
    const sections = document.querySelectorAll('.section-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // --- Demo Inference Logic ---
    const runBtn = document.getElementById('run-inference-btn');
    const loadingContainer = document.getElementById('loading-container');
    const progressBar = document.getElementById('demo-progress');
    const steps = [
        document.getElementById('step-1'),
        document.getElementById('step-2'),
        document.getElementById('step-3'),
        document.getElementById('step-4')
    ];

    const demoPlaceholder = document.getElementById('demo-placeholder');
    const demoOutput = document.getElementById('demo-output');
    const finalScore = document.getElementById('final-score');
    const finalVerdict = document.getElementById('final-verdict');
    const humanSelect = document.getElementById('human-select');

    runBtn.addEventListener('click', () => {
        // Reset state
        runBtn.disabled = true;
        demoPlaceholder.classList.add('hidden');
        demoOutput.classList.add('hidden');
        loadingContainer.classList.remove('hidden');

        steps.forEach(step => {
            step.className = 'step pending';
            const icon = step.querySelector('i');
            icon.className = 'fa-solid fa-circle-notch fa-spin';
        });
        progressBar.style.width = '0%';

        let currentStep = 0;

        const advanceStep = () => {
            if (currentStep > 0) {
                steps[currentStep - 1].className = 'step done';
                const prevIcon = steps[currentStep - 1].querySelector('i');
                prevIcon.className = 'fa-solid fa-check-circle';
            }

            if (currentStep < steps.length) {
                steps[currentStep].className = 'step active';
                progressBar.style.width = `${(currentStep + 1) * 25}%`;
                currentStep++;
                setTimeout(advanceStep, Math.random() * 900 + 600);
            } else {
                setTimeout(() => {
                    showResults();
                    runBtn.disabled = false;
                }, 500);
            }
        };

        advanceStep();
    });

    function showResults() {
        loadingContainer.classList.add('hidden');
        demoOutput.classList.remove('hidden');

        const selectedHuman = humanSelect.value;

        if (selectedHuman === 'Lysozyme') {
            finalScore.textContent = '-0.690';
            finalScore.style.color = '#e74c3c'; // red
            finalScore.style.textShadow = '0 0 20px rgba(231,76,60,0.4)';
            finalVerdict.textContent = '✗ REJECTED (DECOY)';
            finalVerdict.style.background = 'rgba(231,76,60,0.1)';
            finalVerdict.style.color = '#e74c3c';
        } else if (selectedHuman === 'FADD') {
            finalScore.textContent = '+1.000';
            finalScore.style.color = '#50e3c2'; // cyan
            finalScore.style.textShadow = '0 0 20px rgba(80,227,194,0.4)';
            finalVerdict.textContent = '★ STRONG MIMICRY';
            finalVerdict.style.background = 'rgba(80,227,194,0.1)';
            finalVerdict.style.color = '#50e3c2';
        } else {
            // Cyclin D2 default
            finalScore.textContent = '+0.999';
            finalScore.style.color = '#50e3c2';
            finalScore.style.textShadow = '0 0 20px rgba(80,227,194,0.4)';
            finalVerdict.textContent = '★ STRONG MIMICRY';
            finalVerdict.style.background = 'rgba(80,227,194,0.1)';
            finalVerdict.style.color = '#50e3c2';
        }
    }
});
