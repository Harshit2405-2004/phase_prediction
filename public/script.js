document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const errorText = document.getElementById('error-text');
    const loader = document.getElementById('loader');
    const predictBtn = document.getElementById('predict-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loader and hide previous results
        loader.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        errorText.classList.add('hidden');
        predictBtn.disabled = true;
        predictBtn.textContent = 'Predicting...';

        const formData = new FormData(form);
        const compositionStr = formData.get('composition');
        const temperature = formData.get('temperature') ? parseFloat(formData.get('temperature')) : null;
        const route = formData.get('route');

        let compositionDict = {};
        try {
            compositionStr.split(',').forEach(part => {
                const [element, percentage] = part.split(':');
                if (!element || isNaN(parseFloat(percentage))) {
                    throw new Error("Invalid composition format.");
                }
                compositionDict[element.trim()] = parseFloat(percentage);
            });
        } catch (error) {
            displayError('Invalid composition format. Please use "Element:Percent, ...".');
            resetButton();
            return;
        }

        const payload = {
            composition: compositionDict,
            temperature: temperature,
            route: route
        };

        try {
            // *** UPDATED FOR NETLIFY ***
            // The fetch request now points to the Netlify serverless function path.
            const response = await fetch('/.netlify/functions/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'An unknown error occurred.');
            }

            displayResult(data.predicted_phase);

        } catch (error) {
            displayError(error.message);
        } finally {
            resetButton();
        }
    });

    function displayResult(phase) {
        loader.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        resultText.textContent = phase;
        errorText.classList.add('hidden');
    }

    function displayError(message) {
        loader.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        resultText.textContent = '';
        errorText.textContent = `Error: ${message}`;
        errorText.classList.remove('hidden');
    }
    
    function resetButton() {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict Phase';
    }
});
