const PIPELINE = [
    { name: "spritesheet", label: "Build sheet", needsFiles: true},
    { name: "captioner",   label: "Generate caption"},
    { name: "diffusion",   label: "Generate image"}
]

// ====== CACHE VARIABLES ======
let jobId;

// ====== ELEMENTS ========
const dropzones = document.querySelectorAll(".dropzone");
const runButtonEl = document.getElementById('run-button');
const image = document.getElementById('drag-img');

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

// ========= ELEMENTS EVENTS =============
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
        img._file = file; // Keep the image bytes
        img.onload = () => URL.revokeObjectURL(img.src);
        
        zone.appendChild(img);

        enableExecution();
    });
});

runButtonEl.addEventListener('click', async () => {
    
    const form = new FormData();
    // ----- 1. Build the Spritesheet -------
    // 1.1 Build content
    dropzones.forEach(dropzone => {
        const posSectionEl = dropzone.querySelector('img');
        if (posSectionEl._file) form.append(dropzone.id, posSectionEl._file, posSectionEl._file.name)
    })
    await fetch('/api/save/build_spritesheet', {
        method: "POST",
        body: form
    });

})
// ===================================
// ======= BACKEND COM ===============
async function ensureJobData() {

    const resumed = await resume()
    if (resumed) return resumed;

    // if there is no job data available, create one
    const r = await fetch('/api/jobs', { method: 'POST' });
    if (!r.ok) throw new Error('could not create job');

    const { job_id: jobId} = await r.json()
    localStorage.setItem('jobId', jobId);
    return { jobId, step: 0, done: [], artifacts: {} };

}

async function resume() {

    const jobId =  localStorage.getItem('jobId'); 
    if (!jobId) return null;

    const res = await fetch(`/api/jobs/${jobId}/state`);
    if (!res.ok) {
        if (res.status === 404) {            // server wiped it
            localStorage.removeItem('jobId');
        }
        return null;
    }
    return { jobId, ...(await res.json()) };
}

// ====== ON PAGE LOAD =====
jobId, jobData = await ensureJobData()

enableExecution()