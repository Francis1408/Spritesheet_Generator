const image = document.getElementById('drag-img');
const dropzones = document.querySelectorAll(".dropzone");
const runButtonEl = document.getElementById('run-button');


window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());

function enableExecution() {

    let dropzones_filled = 0;
    // Check if three positions are filled
    dropzones.forEach(dropzone => {
        if(dropzone.querySelector('img')) dropzones_filled++;
    })

    if (dropzones_filled >= 3) runButtonEl.disabled = false;
    else runButtonEl.disabled = true;
}


dropzones.forEach(zone => {
    // 2. Drag Over: Necessary to allow dropping
    zone.addEventListener('dragover', (e) => {
        e.preventDefault(); 
        e.dataTransfer.dropEffect = 'copy';
        zone.classList.add('hover');
    });

    // 3. Drag Leave: Visual cleanup when moving away
    zone.addEventListener('dragleave', (e) => {
        if (!zone.contains(e.relatedTarget)) zone.classList.remove('hover');
    });

    // 4. Drop: Move the image element to the new zone
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('hover');

        const file = e.dataTransfer.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            console.warn('Not an image:', file.type);
            return;
        }
        
        // Clear the default text placeholder if it exists
        const placeholder = zone.querySelector('p');
        if (placeholder) placeholder.remove();

        // Clear the current image if exists
        const currentImg = zone.querySelector('img')
        if (currentImg) {
            URL.revokeObjectURL(currentImg.src);
            currentImg.remove();
        }

        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.draggable = true;
        img.onload = () => URL.revokeObjectURL(img.src);
        
        zone.appendChild(img);
        enableExecution();
    });
});

enableExecution()