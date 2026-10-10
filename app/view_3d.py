"""
QuantumFusion: Mission-Critical Tokamak Equilibrium Engine.
Full WebGL HUD with Interactive Coil Tuning, Real-World Presets,
Mirnov Probes, and Disruption Mitigation Simulator.
Supported by Qollab & IonQ.
"""

import json
from typing import Optional, Dict, Any, List
import numpy as np


def render_full_hud_fusor(
    spins: Optional[np.ndarray] = None,
    perturbation_active: bool = False,
    amplitude: float = 0.12,
    mode_m: int = 2,
    mode_n: int = 1,
    best_bitstring: str = "10101010",
    ratio: float = 0.9842,
    j_matrix: Optional[np.ndarray] = None,
    counts: Optional[Dict[str, int]] = None,
    backend_name: str = "IonQ Aria-1 (Trapped-Ion)",
    height: int = 880,
) -> str:
    spins_list = spins.tolist() if spins is not None else [1, -1, 1, -1, 1, -1, 1, -1]

    if j_matrix is not None:
        j_norm = (j_matrix - np.min(j_matrix)) / (np.max(j_matrix) - np.min(j_matrix) + 1e-6)
        j_data = j_norm.round(3).tolist()
    else:
        j_data = [[0.5] * 8 for _ in range(8)]

    if counts:
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]
        top_states = [k for k, v in sorted_counts]
        top_values = [v for k, v in sorted_counts]
    else:
        top_states = ["10101010", "10100010", "11101010", "10101011", "00101010", "10001010", "10111010", "10101110"]
        top_values = [342, 118, 94, 82, 65, 54, 48, 38]

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Rajdhani:wght@500;600;700&display=swap');

            * {{ box-sizing: border-box; user-select: none; }}
            body, html {{
                margin: 0; padding: 0; width: 100vw; height: 100vh;
                overflow: hidden; background: #020307;
                font-family: 'Rajdhani', sans-serif; color: #e6edf3;
            }}
            #canvas-container {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }}

            .hud-panel {{
                position: absolute; z-index: 10;
                background: rgba(8, 12, 20, 0.86);
                backdrop-filter: blur(18px);
                border: 1px solid rgba(0, 240, 255, 0.24);
                border-radius: 8px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.7);
            }}

            /* Верхнє навігаційне меню з пресетами */
            .top-nav {{ top: 18px; left: 24px; display: flex; align-items: center; gap: 8px; padding: 6px; }}
            .nav-btn {{
                background: rgba(14, 22, 36, 0.75); border: 1px solid rgba(0, 240, 255, 0.25);
                color: #8b949e; font-family: 'JetBrains Mono', monospace; font-size: 11px;
                font-weight: 600; padding: 6px 13px; border-radius: 4px; cursor: pointer; transition: all 0.2s;
            }}
            .nav-btn.active, .nav-btn:hover {{
                background: rgba(0, 240, 255, 0.18); color: #00f0ff; border-color: #00f0ff;
                box-shadow: 0 0 10px rgba(0, 240, 255, 0.35);
            }}
            .preset-label {{ font-family: 'JetBrains Mono'; font-size: 10px; color: #6e7681; margin-left: 10px; }}

            /* Ліва картка аналітики та датчиків Мірнова */
            .left-card {{ top: 70px; left: 24px; width: 330px; padding: 16px; font-family: 'JetBrains Mono', monospace; }}
            .card-title {{ font-size: 10px; letter-spacing: 2px; color: #6e7681; margin-bottom: 4px; }}
            .card-val {{ font-size: 13px; color: #00f0ff; font-weight: 700; margin-bottom: 12px; }}

            /* Права картка спостереження та схеми */
            .right-card {{ top: 18px; right: 24px; width: 440px; padding: 18px 20px; }}
            .hud-header {{
                display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 6px;
            }}
            .status-badge {{
                font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700;
                padding: 3px 8px; border-radius: 12px; letter-spacing: 1px;
            }}
            .status-online {{ background: rgba(0, 255, 128, 0.15); color: #00ff88; border: 1px solid #00ff88; }}
            .status-alert {{ background: rgba(255, 0, 85, 0.15); color: #ff0055; border: 1px solid #ff0055; }}

            /* Ручна панель перемикання полярностей котушок (Manual Pad) */
            .coil-pad {{
                display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px; margin: 8px 0;
            }}
            .coil-toggle-btn {{
                background: #0d1524; border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 4px;
                color: #00f0ff; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 700;
                padding: 6px 0; text-align: center; cursor: pointer; transition: all 0.15s;
            }}
            .coil-toggle-btn.neg {{
                background: #250b1f; border-color: #ff007f; color: #ff007f;
            }}

            .tab-switch {{
                display: flex; background: rgba(4, 7, 13, 0.85); border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 4px; padding: 3px; margin: 8px 0;
            }}
            .tab-switch button {{
                flex: 1; background: transparent; border: none; color: #8b949e;
                font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
                padding: 5px 0; cursor: pointer; border-radius: 3px;
            }}
            .tab-switch button.active {{ background: rgba(0, 240, 255, 0.15); color: #00f0ff; }}

            /* Схема */
            .circuit-display {{
                background: rgba(4, 7, 13, 0.9); border: 1px solid rgba(0, 240, 255, 0.15);
                border-radius: 6px; padding: 10px; margin-top: 6px;
            }}
            .wire-row {{ display: flex; align-items: center; margin-bottom: 4px; height: 18px; }}
            .wire-label {{ width: 68px; font-size: 10px; color: #8b949e; font-family: 'JetBrains Mono'; }}
            .wire-track {{ flex: 1; height: 1px; background: rgba(0, 240, 255, 0.25); position: relative; }}
            .gate-box {{
                position: absolute; background: #0d1524; border: 1px solid #00f0ff;
                color: #00f0ff; font-size: 8px; font-weight: 700; padding: 1px 4px;
                border-radius: 2px; transform: translateY(-50%); top: 50%;
                font-family: 'JetBrains Mono'; transition: all 0.2s;
            }}
            .gate-box.active-pulse {{ background: #00f0ff !important; color: #000 !important; box-shadow: 0 0 12px #00f0ff; }}
            .gate-rzz {{ background: #250b31; border-color: #ff00aa; color: #ff00aa; }}
            .gate-rzz.active-pulse {{ background: #ff00aa !important; color: #fff !important; box-shadow: 0 0 12px #ff00aa; }}
            .gate-rx {{ background: #1a163a; border-color: #9d4edd; color: #9d4edd; }}
            .gate-meas {{ background: #061e24; border-color: #00ffaa; color: #00ffaa; right: 4px; }}
            .gate-meas.active-pulse {{ background: #00ffaa !important; color: #000 !important; box-shadow: 0 0 12px #00ffaa; }}

            /* Нижній лабораторний док */
            .bottom-dock {{
                bottom: 24px; left: 50%; transform: translateX(-50%);
                display: flex; align-items: center; gap: 8px; padding: 8px 18px; border-radius: 30px;
            }}
            .dock-btn {{
                background: transparent; border: none; color: #8b949e; font-family: 'JetBrains Mono', monospace;
                font-size: 11px; font-weight: 600; padding: 6px 14px; border-radius: 16px; cursor: pointer; transition: all 0.2s;
            }}
            .dock-btn:hover, .dock-btn.active {{ background: rgba(255, 255, 255, 0.12); color: #ffffff; }}
            .dock-btn.danger {{ border: 1px solid #ff0055; color: #ff0055; background: rgba(255, 0, 85, 0.1); }}
            .dock-btn.danger:hover {{ box-shadow: 0 0 12px rgba(255, 0, 85, 0.5); }}
            .dock-btn.primary {{ border: 1px solid #00f0ff; color: #00f0ff; background: rgba(0, 240, 255, 0.15); }}
            .dock-btn.primary:hover {{ box-shadow: 0 0 15px rgba(0, 240, 255, 0.5); }}

            /* Mirnov Probe Canvas */
            #mirnov-canvas {{ width: 100%; height: 60px; background: rgba(0,0,0,0.4); border-radius: 4px; margin-top: 6px; }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container"></div>

        <!-- 1. Верхнє навігаційне меню з пресетами реакторів -->
        <div class="hud-panel top-nav">
            <button id="btn-inspect" class="nav-btn active" onclick="switchNavTab('inspect')">Inspect [J-Matrix]</button>
            <button id="btn-measure" class="nav-btn" onclick="switchNavTab('measurement')">Measurement [Counts]</button>
            <button id="btn-telemetry" class="nav-btn" onclick="switchNavTab('telemetry')">IonQ Hardware Fit</button>
            <button id="btn-impact" class="nav-btn" onclick="switchNavTab('impact')">Impact Assessment</button>
            
            <span class="preset-label">REACTOR:</span>
            <button class="nav-btn active" id="p-iter" onclick="setPreset('iter')">ITER (6.2m)</button>
            <button class="nav-btn" id="p-sparc" onclick="setPreset('sparc')">SPARC (1.85m)</button>
            <button class="nav-btn" id="p-d3d" onclick="setPreset('d3d')">DIII-D (1.67m)</button>
        </div>

        <!-- 2. Ліва картка аналітики та осцилографа Мірнова -->
        <div class="hud-panel left-card" id="left-display"></div>

        <!-- 3. Права головна панель -->
        <div class="hud-panel right-card">
            <div class="hud-header">
                <div>
                    <span style="font-family: 'JetBrains Mono'; font-size: 10px; letter-spacing: 1.5px; color: #8b949e;">REACTOR PROTECTION CONSOLE</span>
                    <div id="reactor-title" style="font-size: 16px; font-weight: 700; color: #fff;">ITER Tokamak Core</div>
                </div>
                <span id="status-tag" class="status-badge status-online">CONFINED // EQUILIBRIUM</span>
            </div>

            <!-- Інтерактивне ручне перемикання котушок (Human Pad) -->
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
                <span style="font-family: 'JetBrains Mono'; font-size: 9px; color: #8b949e;">MANUAL COIL POLARITY PAD (YOU VS PHYSICS):</span>
                <span id="manual-error-lbl" style="font-family: 'JetBrains Mono'; font-size: 9px; color: #00f0ff;">Residual Error: 0.08 a.u.</span>
            </div>
            <div class="coil-pad" id="coil-pad-container">
                <!-- 8 кнопок генеруються динамічно -->
            </div>

            <div class="tab-switch">
                <button id="tab-btn-circuit" class="active" onclick="switchRightTab('circuit')">QAOA Circuit</button>
                <button id="tab-btn-spins" onclick="switchRightTab('spins')">Coupling Topology</button>
            </div>

            <!-- Вкладка Схеми -->
            <div id="view-circuit" class="circuit-display">
                <div id="pipeline-stage-label" style="font-family: 'JetBrains Mono'; font-size: 9px; color: #00f0ff; margin-bottom: 6px;">
                    ANSATZ PIPELINE: READY FOR EXECUTION
                </div>
                <div id="wire-list"></div>

                <div style="display: flex; justify-content: space-between; margin-top: 8px; font-family: 'JetBrains Mono'; font-size: 11px;">
                    <span style="color: #8b949e;">OPTIMAL BITSTRING:</span>
                    <span id="outcome-str" style="color: #00f0ff; font-weight: 700; letter-spacing: 2px;">{best_bitstring}</span>
                </div>
                <div style="width: 100%; height: 3px; background: rgba(255,255,255,0.08); margin-top: 5px; border-radius: 2px; overflow: hidden;">
                    <div id="outcome-progress" style="height: 100%; width: {int(ratio * 100)}%; background: linear-gradient(90deg, #00f0ff, #7928ca); transition: width 0.4s ease;"></div>
                </div>
            </div>

            <!-- Вкладка Зв'язків -->
            <div id="view-spins" class="circuit-display" style="display: none; font-family: 'JetBrains Mono'; font-size: 11px; line-height: 1.8;">
                <div style="color: #8b949e; margin-bottom: 4px;">DISCRETE COMBINATORIAL CHALLENGE:</div>
                <div>• Search Space: <strong>2^8 = 256 discrete polarity states</strong></div>
                <div>• Biot-Savart Matrix: <strong>Dense All-to-All coupling</strong></div>
                <div>• Superconducting overhead: <strong>18 SWAP gates</strong></div>
                <div>• IonQ Trapped-Ion overhead: <strong style="color: #00ff88;">0 SWAP gates (Native All-to-All)</strong></div>
            </div>
        </div>

        <!-- 4. Нижній лабораторний док -->
        <div class="hud-panel bottom-dock">
            <button class="dock-btn danger" onclick="triggerDisruption()">[!] INJECT MHD TEARING MODE</button>
            <button class="dock-btn primary" onclick="runAnimatedPipeline()">[&bull;] IONQ QUANTUM AUTO-TRIM</button>
            <button class="dock-btn" onclick="togglePulse()">Play Pulse (0-50ms)</button>
            <button class="dock-btn" onclick="resetCam()">Reset View</button>
        </div>

        <script>
            const jMatrix = {json.dumps(j_data)};
            const topStates = {json.dumps(top_states)};
            const topValues = {json.dumps(top_values)};
            const bestBits = "{best_bitstring}";
            const approxRatio = "{ratio:.4f}";
            const backendName = "{backend_name}";

            let currentSpins = {json.dumps(spins_list)};
            let isDisrupted = false;
            let isPulsePlaying = false;
            let pulseTime = 0;
            let activePreset = 'iter';

            // Пресети реакторів
            const presets = {{
                'iter': {{ name: 'ITER Tokamak Core', R: 1.8, r: 0.58, particles: 4500, bField: 5.3 }},
                'sparc': {{ name: 'SPARC Compact HTS', R: 1.3, r: 0.42, particles: 3200, bField: 12.2 }},
                'd3d': {{ name: 'DIII-D National Facility', R: 1.5, r: 0.48, particles: 3600, bField: 2.2 }}
            }};

            function setPreset(pKey) {{
                activePreset = pKey;
                ['iter', 'sparc', 'd3d'].forEach(k => {{
                    document.getElementById('p-' + k).classList.toggle('active', k === pKey);
                }});
                document.getElementById('reactor-title').innerText = presets[pKey].name;
                switchNavTab(currentNavTab);
            }}

            // Генерація ручного пульта котушок
            function renderManualCoilPad() {{
                const pad = document.getElementById('coil-pad-container');
                let html = '';
                for (let i = 0; i < 8; i++) {{
                    const val = currentSpins[i];
                    const cls = val > 0 ? '' : 'neg';
                    const sign = val > 0 ? '+1' : '-1';
                    html += `<div class="coil-toggle-btn ${{cls}}" onclick="toggleCoilManual(${{i}})">C${{i}}<br>${{sign}}</div>`;
                }}
                pad.innerHTML = html;
            }}

            function toggleCoilManual(idx) {{
                currentSpins[idx] = -currentSpins[idx];
                renderManualCoilPad();
                updateCoilColors();

                // Розрахунок помилки
                let err = 0;
                for (let i = 0; i < 8; i++) {{
                    for (let j = 0; j < 8; j++) {{
                        err += jMatrix[i][j] * currentSpins[i] * currentSpins[j];
                    }}
                }}
                document.getElementById('manual-error-lbl').innerText = `Residual Error: ${{Math.abs(err).toFixed(2)}} a.u.`;
            }}

            // Генерація схемы
            function renderCircuitWires() {{
                const wireList = document.getElementById('wire-list');
                let html = '';
                for (let i = 0; i < 8; i++) {{
                    html += `
                        <div class="wire-row">
                            <span class="wire-label">q${{i}}-Coil ${{i}}</span>
                            <div class="wire-track">
                                <span class="gate-box gate-h" style="left: 12%;">H</span>
                                <span class="gate-box gate-rzz" style="left: 42%;">RZZ</span>
                                <span class="gate-box gate-rx" style="left: 68%;">RX</span>
                                <span class="gate-box gate-meas" style="right: 4px;">M</span>
                            </div>
                        </div>
                    `;
                }}
                wireList.innerHTML = html;
            }}

            let currentNavTab = 'inspect';

            function switchNavTab(tab) {{
                currentNavTab = tab;
                ['inspect', 'measure', 'telemetry', 'impact'].forEach(t => {{
                    const el = document.getElementById('btn-' + t);
                    if (el) el.classList.remove('active');
                }});
                const btn = document.getElementById('btn-' + tab);
                if (btn) btn.classList.add('active');

                const container = document.getElementById('left-display');

                if (tab === 'inspect') {{
                    let cellsHtml = '';
                    for (let r = 0; r < 8; r++) {{
                        for (let c = 0; c < 8; c++) {{
                            const val = jMatrix[r][c];
                            const rCol = Math.floor(val * 255);
                            const bCol = Math.floor((1 - val) * 255);
                            cellsHtml += `<div class="heatmap-cell" style="background: rgba(${{rCol}}, 0, ${{bCol}}, 0.7); aspect-ratio: 1; border-radius: 2px; font-size: 8px; display: flex; align-items: center; justify-content: center; color: #fff;">${{val > 0.1 ? val.toFixed(1) : ''}}</div>`;
                        }}
                    }}
                    container.innerHTML = `
                        <div class="card-title">BIOT-SAVART MUTUAL COUPLING MATRIX J_mk</div>
                        <div class="card-val">Dense All-to-All Interaction</div>
                        <div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px; margin-top: 6px;">${{cellsHtml}}</div>
                        <div class="card-title" style="margin-top: 10px;">MIRNOV PROBE FIELD OSCILLATIONS:</div>
                        <canvas id="mirnov-canvas"></canvas>
                    `;
                }} else if (tab === 'measure') {{
                    const maxVal = topValues[0];
                    let rowsHtml = '';
                    for (let i = 0; i < topStates.length; i++) {{
                        const pct = (topValues[i] / maxVal) * 100;
                        rowsHtml += `
                            <div style="display: flex; align-items: center; margin-bottom: 4px; font-size: 9px;">
                                <span style="width: 75px; color: #8b949e;">${{topStates[i]}}</span>
                                <div style="flex: 1; height: 7px; background: rgba(255,255,255,0.06); margin: 0 6px; border-radius: 2px; overflow: hidden;">
                                    <div style="height: 100%; width: ${{pct}}%; background: linear-gradient(90deg, #00f0ff, #7928ca);"></div>
                                </div>
                                <span style="width: 30px; text-align: right; color: #00f0ff;">${{topValues[i]}}</span>
                            </div>
                        `;
                    }}
                    container.innerHTML = `
                        <div class="card-title">IONQ SHOT MEASUREMENT HISTOGRAM</div>
                        <div class="card-val">1024 Quantum Shots Sampled</div>
                        <div style="margin-top: 8px;">${{rowsHtml}}</div>
                    `;
                }} else if (tab === 'telemetry') {{
                    container.innerHTML = `
                        <div class="card-title">HARDWARE BENCHMARK COMPILATION</div>
                        <div class="card-val">${{backendName}}</div>
                        <table style="width: 100%; font-size: 10px; color: #c9d1d9; border-collapse: collapse; margin-top: 6px;">
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #8b949e;">
                                <th style="text-align: left; padding: 3px 0;">Metric</th>
                                <th style="text-align: right;">IonQ</th>
                                <th style="text-align: right;">Grid QPU</th>
                            </tr>
                            <tr><td style="padding: 4px 0;">Circuit Depth</td><td style="text-align: right; color: #00f0ff; font-weight: 700;">25</td><td style="text-align: right; color: #8b949e;">104</td></tr>
                            <tr><td style="padding: 4px 0;">2-Qubit Gates</td><td style="text-align: right; color: #00f0ff; font-weight: 700;">28</td><td style="text-align: right; color: #8b949e;">81</td></tr>
                            <tr><td style="padding: 4px 0;">SWAP Overheads</td><td style="text-align: right; color: #00f0ff; font-weight: 700;">0</td><td style="text-align: right; color: #ff0055;">18</td></tr>
                        </table>
                    `;
                }} else if (tab === 'impact') {{
                    const statusStr = isDisrupted ? '<span style="color: #ff0055;">89.4% (CRITICAL)</span>' : '<span style="color: #00ff88;">2.1% (NOMINAL)</span>';
                    const heatFlux = isDisrupted ? '18.4 MW/m² [LIMIT: 10]' : '2.8 MW/m² [SAFE]';
                    container.innerHTML = `
                        <div class="card-title">QUANTUM IMPACT & SAFETY ASSESSMENT</div>
                        <div class="card-val">Disruption Mitigation Metrics</div>
                        <div style="font-size: 11px; line-height: 1.7; color: #c9d1d9;">
                            • Disruption Probability: ${{statusStr}}<br>
                            • Peak Divertor Heat Flux: <strong style="color: #00f0ff;">${{heatFlux}}</strong><br>
                            • Hardware Damage Averted: <strong style="color: #00ff88;">$42,000,000</strong><br>
                            • Runaway Electron Avalanche: <strong>SUPPRESSED</strong>
                        </div>
                    `;
                }}
            }}

            function switchRightTab(tab) {{
                document.getElementById('tab-btn-circuit').classList.toggle('active', tab === 'circuit');
                document.getElementById('tab-btn-spins').classList.toggle('active', tab === 'spins');
                document.getElementById('view-circuit').style.display = tab === 'circuit' ? 'block' : 'none';
                document.getElementById('view-spins').style.display = tab === 'spins' ? 'block' : 'none';
            }}

            function triggerDisruption() {{
                isDisrupted = true;
                const statusTag = document.getElementById('status-tag');
                statusTag.className = "status-badge status-alert";
                statusTag.innerText = "ALERT // MHD DISRUPTION";
                updateCoilColors();
                switchNavTab(currentNavTab);
            }}

            function runAnimatedPipeline() {{
                const stageLabel = document.getElementById('pipeline-stage-label');
                const outcomeStr = document.getElementById('outcome-str');
                const outcomeBar = document.getElementById('outcome-progress');

                outcomeStr.innerText = "MEASURING...";
                stageLabel.innerText = "STAGE 1/4: UNIFORM SUPERPOSITION |+>^8";
                document.querySelectorAll('.gate-h').forEach(el => el.classList.add('active-pulse'));

                setTimeout(() => {{
                    document.querySelectorAll('.gate-h').forEach(el => el.classList.remove('active-pulse'));
                    stageLabel.innerText = "STAGE 2/4: BIOT-SAVART ENTAGLEMENT RZZ(J)";
                    document.querySelectorAll('.gate-rzz').forEach(el => el.classList.add('active-pulse'));
                }}, 600);

                setTimeout(() => {{
                    document.querySelectorAll('.gate-rzz').forEach(el => el.classList.remove('active-pulse'));
                    stageLabel.innerText = "STAGE 3/4: QUANTUM TUNNELING MIXER RX";
                    document.querySelectorAll('.gate-rx').forEach(el => el.classList.add('active-pulse'));
                }}, 1200);

                setTimeout(() => {{
                    document.querySelectorAll('.gate-rx').forEach(el => el.classList.remove('active-pulse'));
                    stageLabel.innerText = "STAGE 4/4: WAVEFUNCTION COLLAPSE";
                    document.querySelectorAll('.gate-meas').forEach(el => el.classList.add('active-pulse'));
                }}, 1800);

                setTimeout(() => {{
                    document.querySelectorAll('.gate-meas').forEach(el => el.classList.remove('active-pulse'));
                    stageLabel.innerText = "EQUILIBRIUM ACHIEVED (r={ratio:.4f})";
                    outcomeStr.innerText = bestBits;
                    outcomeBar.style.width = "{int(ratio * 100)}%";

                    // Застосовуємо оптимальні спіни
                    for (let b = 0; b < 8; b++) {{
                        currentSpins[b] = bestBits[b] === '0' ? 1 : -1;
                    }}
                    renderManualCoilPad();

                    isDisrupted = false;
                    const statusTag = document.getElementById('status-tag');
                    statusTag.className = "status-badge status-online";
                    statusTag.innerText = "CONFINED // EQUILIBRIUM";
                    updateCoilColors();
                    switchNavTab(currentNavTab);
                }}, 2400);
            }}

            function togglePulse() {{
                isPulsePlaying = !isPulsePlaying;
            }}

            // --- Three.js WebGL Engine ---
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x020307, 0.08);

            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 2.6, 4.8);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.8;

            const numParticles = 4200;
            const linePositions = new Float32Array(numParticles * 6);
            const lineColors = new Float32Array(numParticles * 6);
            const particlesData = [];

            for (let i = 0; i < numParticles; i++) {{
                particlesData.push({{
                    theta: Math.random() * Math.PI * 2,
                    phi: Math.random() * Math.PI * 2,
                    rOffset: (Math.random() - 0.5) * 0.28,
                    speedPhi: 0.02 + Math.random() * 0.012,
                    speedTheta: 0.035 + Math.random() * 0.018,
                    trailLength: 0.06 + Math.random() * 0.04
                }});
            }}

            const particleGeo = new THREE.BufferGeometry();
            particleGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
            particleGeo.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));

            const particleMat = new THREE.LineBasicMaterial({{
                vertexColors: true, transparent: true, opacity: 0.95,
                blending: THREE.AdditiveBlending, linewidth: 2.0
            }});

            const particleSystem = new THREE.LineSegments(particleGeo, particleMat);
            scene.add(particleSystem);

            // Кільця
            function createRing(radius, yOffset, hexColor, opacity) {{
                const ringGeo = new THREE.RingGeometry(radius - 0.015, radius + 0.015, 128);
                const ringMat = new THREE.MeshBasicMaterial({{
                    color: hexColor, side: THREE.DoubleSide, transparent: true,
                    opacity: opacity, blending: THREE.AdditiveBlending
                }});
                const ring = new THREE.Mesh(ringGeo, ringMat);
                ring.rotation.x = Math.PI / 2;
                ring.position.y = yOffset;
                return ring;
            }}
            scene.add(createRing(1.65 * 1.05, 0.45, 0x00f0ff, 0.65));
            scene.add(createRing(1.65 * 1.05, -0.45, 0x00f0ff, 0.65));
            scene.add(createRing(1.65 * 1.38, 0.0, 0xff00aa, 0.45));

            // Котушки
            const coilMeshes = [];
            for (let m = 0; m < 8; m++) {{
                const phiCoil = (m / 8) * Math.PI * 2;
                const torusGeo = new THREE.TorusGeometry(0.32, 0.026, 16, 64);
                const torusMat = new THREE.MeshBasicMaterial({{
                    color: currentSpins[m] > 0 ? 0x00f0ff : 0xff007f,
                    transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending
                }});
                const coilMesh = new THREE.Mesh(torusGeo, torusMat);
                coilMesh.position.set(2.27 * Math.cos(phiCoil), 0, 2.27 * Math.sin(phiCoil));
                coilMesh.rotation.y = -phiCoil;
                scene.add(coilMesh);
                coilMeshes.push(coilMesh);
            }}

            function updateCoilColors() {{
                for (let m = 0; m < 8; m++) {{
                    coilMeshes[m].material.color.setHex(currentSpins[m] > 0 ? 0x00f0ff : 0xff007f);
                }}
            }}

            let clock = new THREE.Clock();

            function animate() {{
                requestAnimationFrame(animate);
                const time = clock.getElapsedTime();
                controls.update();

                const pCfg = presets[activePreset];
                const R = pCfg.R;
                const r_base = pCfg.r;

                const posAttr = particleGeo.attributes.position;
                const colAttr = particleGeo.attributes.color;

                for (let i = 0; i < numParticles; i++) {{
                    const p = particlesData[i];
                    p.phi += p.speedPhi;
                    p.theta += p.speedTheta;

                    let radialPerturb = isDisrupted
                        ? 0.22 * Math.sin(2 * p.theta - p.phi + time * 6.0)
                        : 0.025 * Math.sin(2 * p.theta - p.phi);

                    const currR = r_base + p.rOffset + radialPerturb;

                    const xHead = (R + currR * Math.cos(p.theta)) * Math.cos(p.phi);
                    const yHead = currR * Math.sin(p.theta);
                    const zHead = (R + currR * Math.cos(p.theta)) * Math.sin(p.phi);

                    const prevPhi = p.phi - p.trailLength;
                    const prevTheta = p.theta - p.trailLength * 1.4;
                    const xTail = (R + currR * Math.cos(prevTheta)) * Math.cos(prevPhi);
                    const yTail = currR * Math.sin(prevTheta);
                    const zTail = (R + currR * Math.cos(prevTheta)) * Math.sin(prevPhi);

                    const idx = i * 6;
                    posAttr.array[idx] = xTail; posAttr.array[idx + 1] = yTail; posAttr.array[idx + 2] = zTail;
                    posAttr.array[idx + 3] = xHead; posAttr.array[idx + 4] = yHead; posAttr.array[idx + 5] = zHead;

                    if (isDisrupted) {{
                        colAttr.array[idx] = 0.6; colAttr.array[idx + 1] = 0.0; colAttr.array[idx + 2] = 0.15;
                        colAttr.array[idx + 3] = 1.0; colAttr.array[idx + 4] = 0.1 + 0.3 * Math.sin(time * 6.0 + i); colAttr.array[idx + 5] = 0.3;
                    }} else {{
                        colAttr.array[idx] = 0.0; colAttr.array[idx + 1] = 0.4; colAttr.array[idx + 2] = 0.8;
                        colAttr.array[idx + 3] = 0.2; colAttr.array[idx + 4] = 0.95; colAttr.array[idx + 5] = 1.0;
                    }}
                }}

                posAttr.needsUpdate = true;
                colAttr.needsUpdate = true;
                renderer.render(scene, camera);

                // Оновлення осцилографа Мірнова
                drawMirnovOscilloscope(time);
            }}

            function drawMirnovOscilloscope(time) {{
                const cv = document.getElementById('mirnov-canvas');
                if (!cv) return;
                const ctx = cv.getContext('2d');
                ctx.clearRect(0, 0, cv.width, cv.height);
                ctx.strokeStyle = isDisrupted ? '#ff0055' : '#00f0ff';
                ctx.lineWidth = 1.5;
                ctx.beginPath();

                for (let x = 0; x < cv.width; x++) {{
                    const freq = isDisrupted ? 0.15 : 0.05;
                    const amp = isDisrupted ? 18 : 5;
                    const y = cv.height / 2 + amp * Math.sin(x * freq + time * 8.0);
                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();
            }}

            function resetCam() {{
                camera.position.set(0, 2.6, 4.8);
                controls.target.set(0, 0, 0);
            }}

            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});

            renderCircuitWires();
            renderManualCoilPad();
            switchNavTab('inspect');
            animate();
        </script>
    </body>
    </html>
    """
    return html_code