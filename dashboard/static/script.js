// Cache to store loaded history data
const historyCache = {};
// Cache to store speedmap images
const speedmapCache = {};

function openTrackWorkoutVideo(raceDate, raceNum) {
    const dateParts = raceDate.split('-');
    const dateFormatted = dateParts.join('');
    const raceNumPadded = raceNum.toString().padStart(2, '0');
    const url = `https://streaminghkjc-a.akamaihd.net/hdflash/twstarter/${dateParts[0]}/${dateFormatted}/${raceNumPadded}/novo/twstarter_${dateFormatted}_${raceNumPadded}_novo_2500kbps.mp4`;
    window.open(url, '_blank');
}

function showRace(raceNum) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${raceNum}`).classList.add('active');
    document.querySelectorAll('.race-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`race-${raceNum}`).classList.add('active');
}

function toggleAllRunners(raceNum, buttonElement) {
    const raceContent = document.getElementById(`race-${raceNum}`);
    const runnerRows = raceContent.querySelectorAll('.runner-row');
    const isExpanding = !buttonElement.classList.contains('all-expanded');
    
    runnerRows.forEach(row => {
        const params = row.getAttribute('onclick').match(/'([^']+)'/g).map(s => s.replace(/'/g, ''));
        const horseNo = params[1];
        const historyRow = document.getElementById(`history-${raceNum}-${horseNo}`);
        const cacheKey = `${raceNum}-${horseNo}`;
        
        if (isExpanding) {
            historyRow.classList.add('expanded');
            row.querySelector('.expand-icon').classList.add('expanded');
            
            if (!historyCache[cacheKey]) {
                const historyContent = document.getElementById(`history-content-${raceNum}-${horseNo}`);
                historyContent.innerHTML = '<div class="history-loading">Loading history...</div>';
                const url = `/horse_info/${horseNo}?going=${encodeURIComponent(params[2])}&track=${encodeURIComponent(params[3])}&dist=${encodeURIComponent(params[4])}`;
                
                fetch(url)
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                        return response.text();
                    })
                    .then(html => {
                        historyCache[cacheKey] = html;
                        historyContent.innerHTML = html;
                    })
                    .catch(error => {
                        console.error('Error loading horse history:', error);
                        historyContent.innerHTML = `<div class="history-loading" style="color: #ff4444;">Error loading history: ${error.message}</div>`;
                    });
            } else {
                document.getElementById(`history-content-${raceNum}-${horseNo}`).innerHTML = historyCache[cacheKey];
            }
        } else {
            historyRow.classList.remove('expanded');
            row.querySelector('.expand-icon').classList.remove('expanded');
        }
    });
    
    buttonElement.classList.toggle('all-expanded', isExpanding);
    buttonElement.querySelector('.expand-all-text').textContent = isExpanding ? 'Collapse All' : 'Expand All';
}

function toggleRunnerHistory(rowElement, raceNum, horseNo, going, track, distance) {
    const historyRow = document.getElementById(`history-${raceNum}-${horseNo}`);
    const isExpanded = historyRow.classList.contains('expanded');

    if (isExpanded) {
        historyRow.classList.remove('expanded');
        rowElement.querySelector('.expand-icon').classList.remove('expanded');
    } else {
        historyRow.classList.add('expanded');
        rowElement.querySelector('.expand-icon').classList.add('expanded');

        const cacheKey = `${raceNum}-${horseNo}`;
        if (!historyCache[cacheKey]) {
            const historyContent = document.getElementById(`history-content-${raceNum}-${horseNo}`);
            historyContent.innerHTML = '<div class="history-loading">Loading history...</div>';
            const url = `/horse_info/${horseNo}?going=${encodeURIComponent(going)}&track=${encodeURIComponent(track)}&dist=${encodeURIComponent(distance)}`;

            fetch(url)
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.text();
                })
                .then(html => {
                    historyCache[cacheKey] = html;
                    historyContent.innerHTML = html;
                })
                .catch(error => {
                    console.error('Error loading horse history:', error);
                    historyContent.innerHTML = `<div class="history-loading" style="color: #ff4444;">Error loading history: ${error.message}</div>`;
                });
        } else {
            document.getElementById(`history-content-${raceNum}-${horseNo}`).innerHTML = historyCache[cacheKey];
        }
    }

    updateExpandAllButtonState(raceNum);
}

function updateExpandAllButtonState(raceNum) {
    const raceContent = document.getElementById(`race-${raceNum}`);
    const expandAllBtn = raceContent.querySelector('.expand-all-btn');
    const historyRows = Array.from(raceContent.querySelectorAll('.history-row'));
    const allExpanded = historyRows.length > 0 && historyRows.every(row => row.classList.contains('expanded'));
    
    expandAllBtn.classList.toggle('all-expanded', allExpanded);
    expandAllBtn.querySelector('.expand-all-text').textContent = allExpanded ? 'Collapse All' : 'Expand All';
}

function loadSpeedmap(raceNum) {
    const speedmapImg = document.getElementById(`speedmap-${raceNum}`);
    if (!speedmapImg) return;
    
    if (speedmapCache[raceNum]) {
        speedmapImg.src = speedmapCache[raceNum];
        speedmapImg.style.display = 'block';
        return;
    }
    
    speedmapImg.style.display = 'none';
    
    fetch(`/speedmap/${raceNum}`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.text();
        })
        .then(base64String => {
            speedmapCache[raceNum] = base64String;
            speedmapImg.src = base64String;
            speedmapImg.style.display = 'block';
        })
        .catch(error => {
            console.error(`Error loading speedmap for race ${raceNum}:`, error);
            speedmapImg.style.display = 'none';
        });
}

function loadAllSpeedmaps() {
    document.querySelectorAll('.speedmap-image').forEach(img => {
        loadSpeedmap(img.id.replace('speedmap-', ''));
    });
}

document.addEventListener('DOMContentLoaded', loadAllSpeedmaps);

function sortTable(raceNum, columnIndex, type) {
    const table = document.querySelector(`#race-${raceNum} .runners-table`);
    const tbody = table.querySelector('tbody');
    const header = table.querySelectorAll('thead th')[columnIndex];
    const allHeaders = table.querySelectorAll('thead th.sortable');
    const rows = Array.from(tbody.querySelectorAll('tr')).filter((row, index) => index % 2 === 0);
    const isAscending = !header.classList.contains('sort-asc');
    
    allHeaders.forEach(h => {
        h.classList.remove('sort-asc', 'sort-desc');
        const arrow = h.querySelector('.sort-arrow');
        if (arrow) arrow.textContent = '⇅';
    });
    
    rows.sort((a, b) => {
        let aComp, bComp;
        
        if (type === 'number') {
            aComp = parseFloat(a.cells[columnIndex].textContent.trim());
            bComp = parseFloat(b.cells[columnIndex].textContent.trim());
            if (isNaN(aComp)) aComp = -Infinity;
            if (isNaN(bComp)) bComp = -Infinity;
        } else {
            aComp = a.cells[columnIndex].textContent.trim().toLowerCase();
            bComp = b.cells[columnIndex].textContent.trim().toLowerCase();
        }
        
        return isAscending ? (aComp > bComp ? 1 : aComp < bComp ? -1 : 0) : (aComp < bComp ? 1 : aComp > bComp ? -1 : 0);
    });
    
    header.classList.add(isAscending ? 'sort-asc' : 'sort-desc');
    const arrow = header.querySelector('.sort-arrow');
    if (arrow) arrow.textContent = isAscending ? '▲' : '▼';
    
    rows.forEach(row => {
        const historyRow = row.nextElementSibling;
        tbody.appendChild(row);
        if (historyRow) tbody.appendChild(historyRow);
    });
}

