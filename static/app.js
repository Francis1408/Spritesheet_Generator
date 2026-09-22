// ====== STATE ======
let PIPELINE = [];      // fetched from the server
let job = null;         // { jobId, done, sources, artifacts }
let rebuildFrom = null; // forces the next run back to an earlier step
let MIN_IMAGES = null;

const RENDERERS = {
    spritesheet:     renderSpriteSheet,
    captioner_vlm:   renderCaptionerVLM,
    captioner_llm:   renderCaptionerLLM,
    diffusion:       renderDiffusion
}

// ====== ELEMENTS ========
const dropzones = document.querySelectorAll(".dropzone");
const runButtonEl = document.getElementById('run-button');
const errorEl = document.getElementById('error');
const restartButtonEl = document.getElementById('restart-button');
const upscaleDropdownEl = document.getElementById('upscale-option');
const removeButtonsEl = document.querySelectorAll(".remove-button");
const diffusionButtonEl = document.getElementById('diffusion-settings');
const closeModalButtonsEl = document.querySelectorAll('.modal-btn');

window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());



// ========= ELEMENTS EVENTS =============
dropzones.forEach(zone => {
    // 1. Drag Over: Necessary to allow dropping
    zone.addEventListener('dragover', (e) => {
        e.preventDefault(); 
        e.dataTransfer.dropEffect = 'copy';
        zone.classList.add('hover');
    });

    // 2. Drag Leave: Visual cleanup when moving away
    zone.addEventListener('dragleave', (e) => {
        if (!zone.contains(e.relatedTarget)) zone.classList.remove('hover');
    });

    // 3. Drop: Move the image element to the new zone
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('hover');

        const file = e.dataTransfer.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            showError(`${file.type || 'that file'} is not an image`);
            return;
        }
        
        // Clear the default text placeholder if it exists
        const placeholder = zone.querySelector('p');
        if (placeholder) placeholder.remove();

        // Clear the current image if exists
        const currentImg = zone.querySelector('img')
        if (currentImg) {
            if (currentImg.src.startsWith('blob:')) URL.revokeObjectURL(currentImg.src);
            currentImg.remove();
        }

        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.draggable = true;
        img._file = file; // Keep the image bytes
        img.onload = () => URL.revokeObjectURL(img.src);
        zone.appendChild(img);

        // Swapping a source after the sheet is built means the sheet is stale.
        if (job?.done.includes('spritesheet')) rebuildFrom = 'spritesheet';

        clearError();
        updateButtons();
        renderDropZoneBoxes();
        handlePipelineEnd();
    });
});

removeButtonsEl.forEach(btn => {

    btn.addEventListener('click', (e) => {

        e.preventDefault();
        e.stopPropagation();

        const zone = btn.closest('.dropzone');
        const img = zone.querySelector('img');
        if (img) {
        if (img.src.startsWith('blob:')) URL.revokeObjectURL(img.src);
        img.remove();
        }

        renderDropZoneBoxes();
        updateButtons();
  });

});

closeModalButtonsEl.forEach(btn => {
    btn.addEventListener('click', () => {

        const parentWindow = btn.closest('.window-modal');
        parentWindow.style.display = 'none';

    })
})

runButtonEl.addEventListener('click', async () => {

    // Retrieves the last step executed
    const step = nextStep(job);
    if (!step) return;

    // Disble remove buttons while a step is running
    document.querySelectorAll('.remove-button')
    .forEach(btn => btn.style.display = 'none');

    runButtonEl.disabled = true;
    restartButtonEl.disabled = true;
    runButtonEl.textContent = 'Working…';
    clearError();
    
    let body = undefined;

    try {
        
        if(step.needsFiles) body = collectDropZoneFiles();
        else if (step.needsDiffusionForms) body = collectDiffusionForms();

      
        const res = await fetch(`/api/jobs/${job.jobId}/${step.name}`, {method: 'POST', body});
        const data = await res.json().catch(() => ({}));

        if (!res.ok || data.status !== 'success') {
            throw new Error(data.message || `${step.label} failed (${res.status})`);
        }

        const fresh = await resume();
        if (!fresh) throw new Error('lost track of this job — reload the page');

        job = fresh;
        rebuildFrom = null;
        render();

    } catch (e) {
        showError(e.message);
        updateButtons();
    }

})

restartButtonEl.addEventListener('click', async () => {
    restart()
})

diffusionButtonEl.addEventListener('click', () => {

    const diffusionSettingsEl = document.getElementById('settings-modal');
    diffusionSettingsEl.style.display = 'flex'
})


// ===================================
// ======= BACKEND COM ===============

function nextStep() {

  if (rebuildFrom) return PIPELINE.find(s => s.name === rebuildFrom) ?? null;
  const done = new Set(job?.done ?? []);
  return PIPELINE.find(s => !done.has(s.name)) ?? null;

}

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

// Get the current step state of the run
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

async function restart() {

    const res = await fetch(`/api/jobs/${job.jobId}/reset`, { method: 'POST' });
    if(!res?.status === "success") {
        showError('Error: Couldnt restart the pipeline')
    }
    job = await resume();
    rebuildFrom = null;
    render()
}

// ====== UTILS ===========
function collectDropZoneFiles() {

    const form = new FormData();

    dropzones.forEach(dropzone => {
        const posSectionEl = dropzone.querySelector('img');
        if (posSectionEl?._file) form.append(dropzone.id, posSectionEl._file, posSectionEl._file.name)
    })

    return form;
}

function collectDiffusionForms() {

    // Get diffusion settings
    const diffusionSettingsFormEl = document.getElementById('diffusion-settings-form');
    const form = new FormData(diffusionSettingsFormEl);

    // Get caption 
    const captionTextEl = document.getElementById('caption-box');
    form.append("caption", captionTextEl.value)

    // Get upscale
    const upscaleValue = document.getElementById("upscale-option").value;
    form.append("upscale", upscaleValue)

    return form;
    
}

function filledZones() {
  return [...dropzones].filter(z => z.querySelector('img')).length;
}

function showError(message) {
  if (errorEl) errorEl.textContent = message;
  else alert(message);
}

function clearError() {
  if (errorEl) errorEl.textContent = '';
}

function makePreview(key, value) {

    const card = document.createElement('article');
    card.className = 'preview';
    card.tabIndex = 0;

    const title = document.createElement('h4');
    title.textContent = key;

    const body = document.createElement('div');
    body.className = 'preview-body';
    body.textContent = toText(value);

    card.append(title, body);
    card.addEventListener('click', () => card.classList.toggle('pinned'));
    return card;
}

function toText(value) {

    if (typeof value === 'string') return value;
    if (Array.isArray(value)) return value.map(toText).join('\n\n');
    if (value && typeof value === 'object') {
        return Object.entries(value)
            .map(([k, v]) => `${k}\n${toText(v)}`) 
            .join('\n\n')
    }
    return String(value);
}

async function loadUpscaleOptions() {

    const res = await fetch("/api/upscale-options");
    if (!res.ok) throw new Error(`Failed to load LoRAs: ${res.status}`);
    
    const optionContent = await res.json()

    // Dropdown reference
    const dropdownEl = document.getElementById("upscale-option")

    console.log(optionContent)

    optionContent.forEach((option) =>  {
        const optionTag = document.createElement('option');
        optionTag.textContent = `${option.upscale}x`;
        optionTag.value = option.upscale;

        dropdownEl.appendChild(optionTag)
    })
    
}

// ======= RENDERING FUNCTIONS ===============

function updateButtons() {
  const step = nextStep();

  restartButtonEl.disabled = false;

  if (!step) {
    runButtonEl.textContent = 'Done';
    runButtonEl.disabled = true;
    delete runButtonEl.dataset.step;
    return;
  }

  runButtonEl.textContent = step.label;
  runButtonEl.dataset.step = step.name;
  // Only the upload step depends on how many zones are filled.
  runButtonEl.disabled = step.needsFiles && filledZones() < MIN_IMAGES;
}


function restoreSources() {

  for (const [pos, file] of Object.entries(job.sources ?? {})) {

    const zone = document.getElementById(pos);
    if (!zone || zone.querySelector('img')) continue;

    zone.querySelector('p')?.remove();

    const img = document.createElement('img');
    img.src = `/api/jobs/${job.jobId}/${file}?t=${Date.now()}`;
    img.draggable = true;
    zone.appendChild(img);

  }
}

async function handlePipelineEnd() {
    const step = nextStep();
    if (step) return;

    // If has ended, handle final actions
    const form = new FormData();
    form.append("output_path", "")

    const res = await fetch(`/api/jobs/${job.jobId}/export`, {method: 'POST', form});
    if(!res?.status === "success") {
        showError('Error: Couldnt restart the pipeline')
    }
}

function renderDropZoneBoxes() {
  const atStart = (job?.done.length ?? 0) === 0;

  dropzones.forEach(dropzone => {
    const hasImage = !!dropzone.querySelector('img');
    const btn = dropzone.querySelector('.remove-button');
    let p = dropzone.querySelector('p');

    if (!hasImage && !p) {
      p = document.createElement('p');
      p.textContent = 'Drop Here';
      dropzone.appendChild(p);
    } else if (hasImage && p) {
      p.remove();
    }

    if (btn) btn.style.display = hasImage && atStart ? 'block' : 'none';
  });
}


// Updates the front-end vision
function render() {

    const done = new Set(job.done)
    const next = nextStep(job);
    
    restoreSources();
    renderDropZoneBoxes()

    for (const step of PIPELINE) {
        
        const el = document.querySelector(`[data-step="${step.name}"]`);
        if (!el) continue

        const status = done.has(step.name) ? 'done'
                 : next?.name === step.name ? 'active'
                 : 'locked';

        el.classList.toggle('done',   status === 'done');
        el.classList.toggle('active', status === 'active');
        el.classList.toggle('locked', status === 'locked');

        RENDERERS[step.name]?.(el, job.artifacts?.[step.name], status);
    }

    updateButtons()
}

// STEP 1 RENDERER
function renderSpriteSheet(el, artifact, status) {

    const output = document.getElementById('output')

    // Clear current image
    const currentImg = output.querySelector('img')
    if (currentImg) {
        if (currentImg.src.startsWith('blob:')) URL.revokeObjectURL(currentImg.src);
        currentImg.remove();
    }

    if(artifact?.url) {
        const img = document.createElement('img');
        img.src = `${artifact.url}?t=${Date.now()}`;
        output.appendChild(img);
    }

    dropzones.forEach(z => z.classList.toggle('frozen', status === 'done'));
}

// STEP 2 RENDERER
function renderCaptionerVLM(el, artifact, status) {

    const host = el.querySelector('.previews');
    host.innerHTML = '';
    if(!artifact?.data) return;

    for (const [key, value] of Object.entries(artifact.data)) {
        host.appendChild(makePreview(key, value))

    }

}

function renderCaptionerLLM(el, artifact, status) {

    const host = el.querySelector('.caption-text');
    host.value = '';
    if (errorEl) errorEl.textContent = '';
    if(!artifact?.data) return;

    const { data: caption, length, exceed } = artifact.data;
    host.value = caption;


    // Place warning if caption exceeds token limit
    if(exceed && errorEl) {
        const over = length - 77;
        errorEl.textContent =
            `WARNING: The caption exceeds the limit by ${over} token${over === 1 ? '' : 's'}.\n` +
            `Edit it for better results.`;

    }

}

function renderDiffusion(el, artifact) {

    const rawOutput = el.querySelector("#output-raw");
    const snappedOutput = el.querySelector("#output-snapped");

    if(!artifact?.data) return;

    // clear previous renders, otherwise images pile up on every render()
    const add = (host, url) => {
        if (!host || !url) return;
        const img = document.createElement('img');
        img.src = `${url}?t=${Date.now()}`;
        host.appendChild(img);
    };

    add(rawOutput, artifact.data.crop_raw);
    add(snappedOutput, artifact.data.crop_snapped);

}

// ====== ON PAGE LOAD ======
async function init() {
  try {
    PIPELINE = await (await fetch('/api/pipeline')).json();
    MIN_IMAGES = (await (await fetch('/api/minimages')).json()).value;
    job = await ensureJobData();
    await loadUpscaleOptions();
    render();
  } catch (e) {
    showError(e.message);
  }
}

init();