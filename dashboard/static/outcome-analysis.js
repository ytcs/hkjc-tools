// HKJC Race Info - Outcome Analysis
const OutcomeAnalysis = {
    state: new Map(),
    
    getState(raceNum) {
        if (!this.state.has(raceNum)) {
            this.state.set(raceNum, {
                collapsed: false,
                k: 100,
                m: 3
            });
        }
        return this.state.get(raceNum);
    },
    
    toggle(raceNum) {
        const validRaceNum = Utils.validateRaceNum(raceNum);
        const state = this.getState(validRaceNum);
        const raceContent = Utils.$(`#race-${validRaceNum}`);
        if (!raceContent) return;
        
        const section = Utils.$('.outcome-analysis-section', raceContent);
        if (!section) return;
        
        const content = Utils.$(`#outcome-content-${validRaceNum}`, section);
        const header = Utils.$('.section-header', section);
        
        if (!content || !header) return;
        
        state.collapsed = !state.collapsed;
        if (state.collapsed) {
            Utils.addClass(content, 'collapsed');
            Utils.addClass(header, 'collapsed');
        } else {
            Utils.removeClass(content, 'collapsed');
            Utils.removeClass(header, 'collapsed');
        }
    },
    
    async load(raceNum) {
        const validRaceNum = Utils.validateRaceNum(raceNum);
        const resultsContainer = Utils.$(`#outcome-results-${validRaceNum}`);
        if (!resultsContainer) return;
        
        const kInput = Utils.$(`#outcome-k-${validRaceNum}`);
        const mInput = Utils.$(`#outcome-m-${validRaceNum}`);
        
        if (!kInput || !mInput) return;
        
        const k = parseInt(kInput.value) || 100;
        const m = parseInt(mInput.value) || 3;
        
        if (k < 1 || k > 1000 || m < 1 || m > 10) {
            resultsContainer.innerHTML = '<div class="outcome-error">Invalid parameters. k must be 1-1000, m must be 1-10.</div>';
            return;
        }
        
        resultsContainer.innerHTML = '<div class="outcome-loading">Loading outcome analysis...</div>';
        
        try {
            const data = await API.fetchOutcomes(validRaceNum, k, m);
            this.display(validRaceNum, data, m);
        } catch (error) {
            console.error('Error loading outcome analysis:', error);
            resultsContainer.innerHTML = '<div class="outcome-error">Error loading outcome analysis. Please try again.</div>';
        }
    },
    
    display(raceNum, data, m) {
        const resultsContainer = Utils.$(`#outcome-results-${raceNum}`);
        if (!resultsContainer) return;
        
        if (!data || !data.ranking || !data.prob || data.ranking.length === 0) {
            resultsContainer.innerHTML = '<div class="outcome-error">No outcome data available</div>';
            return;
        }
        
        // Calculate probability for each horse to be in the outcome
        const horseProbs = this.calculateHorseProbabilities(data.ranking, data.prob);
        
        // Build probability table
        const probTableHtml = this.buildProbabilityTable(horseProbs);
        
        // Build outcomes table (max 10)
        const outcomesTableHtml = this.buildOutcomesTable(data.ranking, data.prob, m, 10);
        
        resultsContainer.innerHTML = probTableHtml + outcomesTableHtml;
    },
    
    calculateHorseProbabilities(rankings, probs) {
        const horseProbs = new Map();
        
        rankings.forEach((ranking, idx) => {
            const prob = probs[idx] || 0;
            ranking.forEach(horse => {
                const current = horseProbs.get(horse) || 0;
                horseProbs.set(horse, current + prob);
            });
        });
        
        // Convert to array and sort by probability descending
        return Array.from(horseProbs.entries())
            .map(([horse, prob]) => ({ horse, prob }))
            .sort((a, b) => b.prob - a.prob);
    },
    
    buildProbabilityTable(horseProbs) {
        if (horseProbs.length === 0) return '';
        
        const rows = horseProbs.map(({ horse, prob }) => `
            <tr>
                <td><strong>${horse}</strong></td>
                <td class="outcome-probability-value">${(prob * 100).toFixed(2)}%</td>
            </tr>
        `).join('');
        
        return `
            <table class="outcome-probability-table">
                <thead>
                    <tr>
                        <th>Horse</th>
                        <th>Probability in Outcome</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
        `;
    },
    
    buildOutcomesTable(rankings, probs, m, maxRows) {
        const displayCount = Math.min(rankings.length, maxRows);
        if (displayCount === 0) return '';
        
        const headerCells = Array.from({ length: m }, (_, i) => 
            `<th>Position ${i + 1}</th>`
        ).join('');
        
        const rows = rankings.slice(0, displayCount).map((ranking, idx) => {
            const prob = probs[idx] || 0;
            const horseCells = ranking.map(horse => 
                `<td><span class="outcome-horse-badge">${horse}</span></td>`
            ).join('');
            
            return `
                <tr>
                    <td><span class="outcome-prob-badge">${(prob * 100).toFixed(2)}%</span></td>
                    ${horseCells}
                </tr>
            `;
        }).join('');
        
        return `
            <div class="outcome-table-container">
                <table class="outcome-table">
                    <thead>
                        <tr>
                            <th>Probability</th>
                            ${headerCells}
                        </tr>
                    </thead>
                    <tbody>
                        ${rows}
                    </tbody>
                </table>
                ${rankings.length > maxRows ? `<div style="text-align: center; padding: 10px; color: #666; font-size: 0.85em;">Showing top ${maxRows} of ${rankings.length} outcomes</div>` : ''}
            </div>
        `;
    },
    
    initAll() {
        // Outcome analysis sections start expanded by default
        Utils.$$('.outcome-analysis-section').forEach(section => {
            const raceNum = section.closest('.race-content')?.id.replace('race-', '');
            if (raceNum) {
                const state = this.getState(raceNum);
                state.collapsed = false;
            }
        });
    }
};

window.toggleOutcomeAnalysis = (raceNum) => OutcomeAnalysis.toggle(raceNum);
window.loadOutcomeAnalysis = (raceNum) => OutcomeAnalysis.load(raceNum);

