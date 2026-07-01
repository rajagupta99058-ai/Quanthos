import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";
import { getFallbackResponse, classifySemanticQuery, hasManualFallback } from "./src/utils/fallbackTemplates.js";

dotenv.config();

enum ThinkingLevel {
  HIGH = "HIGH",
  MEDIUM = "MEDIUM",
  LOW = "LOW"
}

const app = express();
const PORT = 3000;

app.use(express.json());

// Google Search Console verification endpoint
app.get("/google3962aa20edca8061.html", (req, res) => {
  res.setHeader("Content-Type", "text/html");
  res.send("google-site-verification: google3962aa20edca8061.html");
});

// Simple in-memory tracker for guest trials by IP address
const ipGuestUsesMap = new Map<string, number>();

function getClientIp(req: express.Request): string {
  const forwarded = req.headers["x-forwarded-for"];
  if (forwarded) {
    const ips = typeof forwarded === "string" ? forwarded.split(",") : forwarded;
    if (Array.isArray(ips)) {
      return ips[0].trim();
    }
    return String(forwarded).trim();
  }
  return req.socket.remoteAddress || "unknown-ip";
}

// Endpoint to check guest uses and sync with client
app.post("/api/check-guest-uses", (req, res) => {
  const { localCount } = req.body;
  const ip = getClientIp(req);
  const serverCount = ipGuestUsesMap.get(ip) || 0;
  
  // Take the maximum of local and server to prevent resetting by clearing localStorage
  const finalCount = Math.max(Number(localCount) || 0, serverCount);
  
  // Sync backend map
  ipGuestUsesMap.set(ip, finalCount);
  
  res.json({ count: finalCount });
});

// Endpoint to increment guest uses
app.post("/api/increment-guest-uses", (req, res) => {
  const ip = getClientIp(req);
  const serverCount = ipGuestUsesMap.get(ip) || 0;
  const finalCount = Math.min(2, serverCount + 1);
  
  ipGuestUsesMap.set(ip, finalCount);
  res.json({ count: finalCount });
});

// Initialize server-side Gemini SDK if API Key is available (supports GEMINI_API_KEY, GEMINI_API_KEY_2, and GEMINI_API_KEY_3)
const apiKey = process.env.GEMINI_API_KEY;
const apiKey2 = process.env.GEMINI_API_KEY_2;
const apiKey3 = process.env.GEMINI_API_KEY_3;
const ai = apiKey
  ? new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    })
  : (apiKey2
      ? new GoogleGenAI({
          apiKey: apiKey2,
          httpOptions: {
            headers: {
              "User-Agent": "aistudio-build",
            },
          },
        })
      : (apiKey3
          ? new GoogleGenAI({
              apiKey: apiKey3,
              httpOptions: {
                headers: {
                  "User-Agent": "aistudio-build",
                },
              },
            })
          : null));

const generationCache = new Map<string, any>();

function normalizePrompt(p: string): string {
  return p
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

async function callOpenRouterCodeGenerator(topic: string, visualDesignBlueprint: string, interactiveParameters: any[]): Promise<string> {
  const rawApiKey = process.env.ZAI_API_KEY || process.env.GLM_API_KEY || process.env.OPENROUTER_API_KEY || "";
  let apiKey = rawApiKey.trim();
  if (apiKey.startsWith('"') && apiKey.endsWith('"')) {
    apiKey = apiKey.slice(1, -1);
  }
  if (apiKey.startsWith("'") && apiKey.endsWith("'")) {
    apiKey = apiKey.slice(1, -1);
  }
  apiKey = apiKey.trim();

  if (!apiKey) {
    throw new Error("OpenRouter API key is not configured. Please add ZAI_API_KEY under settings.");
  }

  const systemPrompt = `You are an elite 3D web developer and creative coder.
Your task is to write clean, high-performance, and extremely polished Three.js JavaScript code to visualize a given PCMB concept.

You will receive the topic name, the detailed scientific visual design blueprint, and the interactive parameters designed for this simulation.

You MUST follow these strict guidelines to write the JavaScript code:
1. The code must be self-contained and run inside an evaluation function block.
2. It receives these 4 parameters:
   - "container" (the DOM div element wrapper)
   - "THREE" (the global Three.js library object)
   - "OrbitControls" (from OrbitControls)
   - "currentParams" (the object of initial user-adjustable parameters)
3. The code MUST return an object with:
   - "updateParams(newParams)": Callback to update running parameters (e.g. speed, gravity) without rebuild.
   - "destroy()": Callback to cancel animation requests, dispose of all geometries, materials, textures, and remove any dynamically appended DOM element wrappers from the "container".
4. Background: Always set background to a dark slate/science shade like '#0a0a0f':
   scene.background = new THREE.Color('#0a0a0f');
   scene.fog = new THREE.FogExp2('#0a0a0f', 0.01);
5. Dynamic Floating UI Card: Generate a glassmorphic floating div inside the container to display real-time parameter values and LaTeX-free formulas.
6. Floating 3D/HTML Labels: For key meshes, generate div labels that project their 3D positions onto screen coordinates and update in the animation loop.
7. Cleanup: Rigorously clean up and remove the dynamic card and floating label divs inside the destroy() callback.
8. Anti-Black-Screen Mandate: Use ONLY bright neon colors (e.g. Cyan #00ffd2, Magenta #ff0077, Lime #39ff14, Orange #ffaa00, Yellow #ffff00, Red #ff3366, Electric Blue #0099ff). Set emissive colors and emissiveIntensity (0.5 to 1.0) on materials (MeshStandardMaterial or MeshPhysicalMaterial). Do not use dark colors.
9. Lights: Add AmbientLight('#ffffff', 0.4), DirectionalLight('#ffffff', 1.5), and at least 2 colored PointLights for vibrant neon aesthetics.
10. Animation: Implement physical integration/equations inside requestAnimationFrame. Keep it extremely performant.
11. No external asset loaders (.gltf, .obj, etc.). Generate all textures procedurally using HTML Canvas or built-in geometries.

Generate ONLY the ready-to-evaluate JavaScript code. Do NOT output any markdown tags, markdown blocks, conversational explanations, or introductory text. Just pure JavaScript.`;

  const userPrompt = `Topic: "${topic}"

Scientific Visual Blueprint:
${visualDesignBlueprint}

Interactive Parameters Definition:
${JSON.stringify(interactiveParameters, null, 2)}

Please write the complete, self-contained Three.js JavaScript code based on the instructions. Ensure it has an updateParams and destroy callback, and contains the required floating UI card and labels. Do not wrap the output in markdown code blocks.`;

  const candidateModels = [
    "cohere/north-mini-code",
    "meta-llama/llama-3-8b-instruct:free",
    "cohere/command-r-plus",
    "meta-llama/llama-3.1-8b-instruct:free"
  ];

  let lastError: Error | null = null;

  for (const model of candidateModels) {
    try {
      const url = "https://openrouter.ai/api/v1/chat/completions";
      console.log(`[OpenRouter Hybrid] Requesting Three.js code using model: ${model}`);

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`,
          "HTTP-Referer": "https://ai.studio/build",
          "X-Title": "Quanthos Science Studio"
        },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt }
          ],
          temperature: 0.2
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.warn(`[OpenRouter Hybrid Warning] Model "${model}" failed (Status ${response.status}): ${errorText}`);
        if (response.status === 400 || response.status === 404) {
          lastError = new Error(`Model ${model} failed (Status ${response.status}): ${errorText}`);
          continue;
        }
        throw new Error(`OpenRouter API error (Status ${response.status}): ${errorText}`);
      }

      const data = await response.json();
      const textContent = data.choices?.[0]?.message?.content;
      if (!textContent) {
        throw new Error(`Empty response received from OpenRouter model "${model}".`);
      }

      let cleanedText = textContent.trim();
      if (cleanedText.startsWith("```")) {
        cleanedText = cleanedText.replace(/^```(?:javascript|js)?\s*/i, "").replace(/\s*```$/, "");
      }
      cleanedText = cleanedText.trim();
      
      return cleanedText;
    } catch (err: any) {
      console.error(`[OpenRouter Hybrid Error] Model "${model}" failed:`, err.message);
      lastError = err;
    }
  }

  throw lastError || new Error("All configured candidate OpenRouter models failed.");
}

// Endpoint to explain the science/math concept and generate interactive Three.js rendering code
app.post("/api/explain-and-render", async (req, res) => {
  const { prompt, forceFresh, thinkingMode, isLoggedIn, modelProvider } = req.body;

  if (!prompt || typeof prompt !== "string") {
    return res.status(400).json({ error: "No topic prompt provided." });
  }

  // If the user is not signed in, enforce the 2-trial limit strictly via server-side IP tracking
  if (!isLoggedIn) {
    const ip = getClientIp(req);
    const serverCount = ipGuestUsesMap.get(ip) || 0;
    if (serverCount >= 2) {
      return res.status(403).json({ error: "Aapne apne 2 free trials pure kar liye hain. Kisi bhi concept ke unlimited dynamic 3D simulations ke liye abhi Sign In karein!" });
    }
  }

  const cachedKey = normalizePrompt(prompt);
  if (forceFresh === true) {
    generationCache.delete(cachedKey);
    console.log(`[Cache Purged] Force-clearing cache for key: "${cachedKey}" prior to regeneration.`);
  }

  if (generationCache.has(cachedKey)) {
    console.log(`[Cache Hit] Serving cached simulation for key: "${cachedKey}" (Original: "${prompt}")`);
    return res.json(generationCache.get(cachedKey));
  }

  const hasOpenRouterKey = !!(process.env.ZAI_API_KEY || process.env.GLM_API_KEY || process.env.OPENROUTER_API_KEY);

  if (!ai) {
    if (hasManualFallback(prompt)) {
      // If Gemini API is missing, gracefully return our optimized procedural fallback instead of crashing or returning an error
      const fallbackVal = getFallbackResponse(prompt, "No API Key configured. Loading stable procedural simulation scene.");
      generationCache.set(cachedKey, fallbackVal);
      return res.json(fallbackVal);
    } else {
      return res.status(400).json({ error: "Is topic ke liye abhi simulation generate nahi ho saka, dusra topic try karo ya thodi der baad try karo" });
    }
  }

  try {
    const classification = classifySemanticQuery(prompt);

    // Build a precise structural hint based on the exact query
    const promptLower = prompt.toLowerCase();
    let structuralHint = "";

    if (/graphene/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: Graphene is a single-layer hexagonal honeycomb lattice of carbon atoms. You MUST generate a flat 2D hexagonal grid of carbon atoms (dark grey spheres) connected by covalent bonds (cylinders). Each carbon has exactly 3 neighbors at 120° angles. The lattice constant is ~0.142 nm. Do NOT generate NaCl ionic cubic lattice, do NOT generate any sodium or chloride atoms, do NOT generate a 3D cubic structure. This is purely a 2D carbon sheet with sp2 hybridized bonds.`;
    } else if (/nacl|sodium chloride|salt/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: NaCl is a face-centered cubic ionic lattice alternating Na+ (blue spheres) and Cl- (green spheres). Do NOT generate graphene or carbon structures.`;
    } else if (/dna|double helix/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: DNA is a double helix — two antiparallel strands of nucleotides wound around each other. Generate two helical backbone curves with horizontal base pair rungs (A-T in red/blue, G-C in green/yellow) connecting them. Do NOT generate any lattice or crystal structure.`;
    } else if (/benzene/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: Benzene (C6H6) is a planar hexagonal ring of 6 carbon atoms with alternating double bonds, each carbon bonded to one hydrogen. Generate exactly this flat hexagonal ring. Do NOT generate a lattice or 3D crystal.`;
    } else if (/water|h2o/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: Water (H2O) is a bent molecule — one large red oxygen atom bonded to two smaller white hydrogen atoms at a 104.5° angle. Generate exactly this molecular geometry.`;
    } else if (/diamond/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: Diamond is a 3D tetrahedral carbon lattice where each carbon bonds to exactly 4 neighbors in a tetrahedral arrangement (109.5°). Generate this sp3 tetrahedral cubic structure. Do NOT generate graphene hexagonal lattice.`;
    } else if (/neuron/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: A neuron has a central soma (cell body), multiple branching dendrites, a long axon with myelin sheath segments, and axon terminals. Generate this biological structure only. Do NOT generate any crystal or molecular lattice.`;
    } else if (/mitochondria/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: A mitochondrion has an outer membrane (smooth oval shell), inner membrane with cristae folds (accordion-like internal folds), and a matrix interior. Generate this organelle. Do NOT generate any crystal or lattice.`;
    } else if (/lorenz|lorenz attractor/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: The Lorenz Attractor is a chaotic trajectory traced by integrating dx/dt=σ(y-x), dy/dt=x(ρ-z)-y, dz/dt=xy-βz with σ=10, ρ=28, β=8/3. Generate a tube/line trail of this butterfly-shaped trajectory. Do NOT generate any molecular or biological structure.`;
    } else if (/black hole/i.test(promptLower)) {
      structuralHint = `MANDATORY STRUCTURE FOR THIS QUERY: A black hole visualization should show a dark central singularity sphere, an accretion disk of glowing particles orbiting around it, and gravitational lensing light bending effect. Use particle systems for the disk. Do NOT generate molecular or biological structures.`;
    }

    const systemPrompt = `You are an elite 3D web developer, physicist, mathematician, chemist, and biologist.
Your task is to take the user's exact science, mathematical, chemical, biological, or medical topic query: "${prompt}".
You MUST render exactly this query, and provide highly polished, interactive Three.js JavaScript code to render a gorgeous 3D scene of that exact concept.

⚠️ CONCISE MULTI-SECTION GENERATION DIRECTIVE:
1. Keep the generated Three.js code in "threeJsCode" highly optimized, lightweight, modular, and strictly under 120 lines of code.
2. In the "explanation" field, generate a concise Markdown text containing exactly these three sub-sections:
   - ### Scientific Grounding: Short bullet points and key formulas (using standard LaTeX formulas wrapped in $$ or $).
   - ### Step-by-Step Derivation: Short bullet points and key formulas (using standard LaTeX formulas wrapped in $$ or $).
   - ### Practice Questions: Exactly 2 sweet, brief, and direct practice questions with their solutions.
3. Keep all of these explanation sections very short, sweet, and direct to prevent model timeouts or exceeding token limits.
4. Keep "controlsGuide" very brief (1-2 sentences on how to interact).
5. You MUST populate the "theoryData" object with ALL four of the following fields — no field may be omitted or left empty:
   - "conceptAndFormula": A concise explanation of the core concept and its primary formula(s). Wrap all math in $$ for display blocks or $ for inline. Use LaTeX for every equation.
   - "derivation": A numbered step-by-step mathematical derivation of the concept. Use LaTeX for every equation (wrapped in $$ or $).
   - "realLifeApplications": Exactly 2 to 3 real-world application strings (plain text sentences) describing how this concept is used in practice.
   - "practiceQuestions": Exactly 2 objects each with a "question" string and an "answer" string. Keep both concise and direct.

${structuralHint ? `⚠️ CRITICAL STRUCTURE OVERRIDE — READ THIS FIRST:\n${structuralHint}\n` : ""}

CRITICAL CLASSIFICATION PARADIGM DIRECTIVE:
We have automatically classified this search query as belonging to the semantic class: **${classification}**.
You MUST strictly align the visual structure and rendering design of your returned Three.js code to the rules of this class:
1. CONST_ORGANIC: Biology, cells, blood, organelles. You must strictly generate grouped organic geometries (smooth spheres/blobs, high roughness, flesh/tissue tones). You MUST completely hide default wireframes, and you MUST completely hide floor grids or coordinate axes helpers. Maintain an organic, wet-lab/microscope bioluminescent dark environment.
2. CONST_PARTICLE_CLOUD: Space, nebula, fluids, clusters, gas. You must render point-based vertex clouds using Points and procedurally generated soft radial glow textures.
3. CONST_MOLECULAR: Atoms, bonds, crystal lattices, catalysts. You must render standard CPK ball-and-stick connectors (spheres + cylinders) to depict compounds and chemical networks.
4. CONST_MATHEMATICAL: Wave formulas, coordinate manifolds, vector grids, chaos attractors. You must render explicit coordinate geometries, function manifolds, grid/polar helpers, and color-coded Cartesian coordinate axes helpers.

CRITICAL DIRECTIVE:
1. STRICT ADHERENCE TO THE QUERY: You must strictly parse the exact search query "${prompt}" instead of returning a generic topic from that subject category. If the query is "Graphene", you MUST ONLY generate a hexagonal carbon honeycomb lattice — NOT NaCl, NOT cubic lattice, NOT any other structure. If the query is "DNA", you MUST only generate a 3D DNA Double Helix model. If the query is "Neuron", you must generate a 3D neuron cell. NEVER substitute one structure for another.
2. NO GENERIC REUSE OR PRE-BAKED RESPONSES: Do not reuse standard preset scenes or default patterns unless the query matches them exactly verbatim. Do not return any generic box/sphere-on-a-grid fallback. Do NOT use NaCl as a generic fallback for molecular queries.
3. CUSTOM-TAILORED ENGINEERING: The generated Three.js code and explanation must be explicitly and dynamically engineered around the specific keywords provided in "${prompt}". Create interactive parameters, structures, colors, and dynamic particles that directly represent the actual mechanisms of "${prompt}".
4. SELF-CHECK BEFORE RESPONDING: Before returning your code, verify: Does my generated structure actually match "${prompt}"? Are my atom labels, bond geometry, and structure topology correct for "${prompt}"? If not, regenerate.

The code MUST be self-contained within a function body. It receives:
1. "container" (the DOM div element wrapper)
2. "THREE" (the global Three.js library object)
3. "OrbitControls" (from OrbitControls)
4. "currentParams" (the object of initial user-adjustable state variables)

And the code MUST return an object of the signature:
{
  updateParams: (newParams) => { /* Update the running simulation parameters (e.g. speed, frequency, gravity) without rebuilding the whole scene */ },
  destroy: () => { /* Rigorously clean up: cancel animation frame requests, remove canvas from container, dispose geometries, materials, textures, lines, lights, event listeners */ }
}

GUIDRULES FOR THE 3D THREEJS CODE:
- Mandatory UI Floating Card: Every single simulation must render a sleek glassmorphic widget floating at the top inside the container canvas element. For example, for an Atomic Lattice, it must show the geometric/diffraction formula (like Bragg's Law 2d \\sin\\theta = n\\lambda or Lattice vectors) with real-time changing parameter values! Create this card dynamically by doing:
  const topCardDiv = document.createElement('div');
  topCardDiv.style.position = 'absolute';
  topCardDiv.style.top = '12px';
  topCardDiv.style.left = '50%';
  topCardDiv.style.transform = 'translateX(-50%)';
  topCardDiv.style.zIndex = '10';
  topCardDiv.style.pointerEvents = 'none';
  container.appendChild(topCardDiv);
  Then, inside the animate loop, update topCardDiv.innerHTML with LaTeX-free formulas and active parameter values, formatted beautifully with:
  "display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(15, 23, 42, 0.55); backdrop-filter: blur(8px); webkit-backdrop-filter: blur(8px); border: 1.5px solid #ff3366; border-radius: 10px; padding: 6px 12px; font-family: Inter, sans-serif; box-shadow: 0 6px 20px rgba(0,0,0,0.4); width: max-content; font-size: 11px; text-shadow: 0 1px 2px rgba(0,0,0,0.8); color: white;"
- Mandatory Floating 3D/HTML Labels: Every active 3D node/atom/object (e.g., 'Atom A', 'Node Position', 'Bob 1', 'Bob 2', etc.) MUST have floating HTML tags dynamically attached to them.
  Create HTML tag divs (e.g. const labelDiv = document.createElement('div'); append to container; absolute position, transition, pointer-events: none, blur backdrop, thin neat colored border matching the mesh color, clear white text shadow).
  Inside the animate loop, project their 3D coordinates onto screen pixels using:
  const tempV = new THREE.Vector3();
  tempV.copy(activeMesh.position);
  tempV.project(camera);
  const widthHalf = container.clientWidth / 2;
  const heightHalf = container.clientHeight / 2;
  if (tempV.z > 1) {
    labelDiv.style.opacity = '0';
  } else {
    labelDiv.style.opacity = '1';
    const lx = (tempV.x * widthHalf) + widthHalf;
    const ly = -(tempV.y * heightHalf) + heightHalf;
    labelDiv.style.left = lx + 'px';
    labelDiv.style.top = (ly - 15) + 'px'; // offset slightly above
    labelDiv.innerHTML = '<div style="...">Label Name: (' + activeMesh.position.x.toFixed(1) + ', ' + activeMesh.position.y.toFixed(1) + ', ' + activeMesh.position.z.toFixed(1) + ')</div>';
  }
- Rigorous DOM Cleanup: In the return "destroy" callback, you MUST target and remove all dynamically appended label divs and the topmost card div using parentNode.removeChild, preventing lingering orphans during route or topic changes.
- Styling: Do NOT use plain boring spheres on black backgrounds. Apply dramatic lights (directional lights, high-contrast spotlighting, point lights with neon colors), beautiful grid systems (GridHelper, PolarGridHelper), beautiful particle systems (Points, PointsMaterial), trailing effects (using dynamic buffers or a series of historical coordinate meshes), and coordinates axes helpers (where appropriate). Use MeshPhysicalMaterial, MeshStandardMaterial, or glassmorphic materials for stunning aesthetics.
- Animations & Physics: Implement actual integration equations (such as Euler or Verlet for gravity, orbits, double pendulums, wave interference, or Lorenz Attractor differential updates). Update the meshes inside an 'requestAnimationFrame' loop.
- Interactivity: Read variables from 'currentParams' initially and inside 'updateParams(newParams)'. For example, if a parameter is 'gravity', bind your internal gravity variable to 'newParams.gravity'. Keep references to mesh positions, frequencies, scales, speeds, and update them smoothly without recreating objects.
- Self-contained asset rules: No external image or 3D model loaders (.gltf, .obj, jpg, png) are permitted, because there are no host assets. Generate all textures procedurally using HTML Canvas (e.g. a glowing particle radial gradient texture) or use Three.js built-in shapes (TorusKnot, Custom Parametric Geometries, Ring, Lenses, etc.).
- Robustness: Put safety checks so the code does not crash if elements are missing. Ensure OrbitControls use the container for DOM events so they do not capture the whole window. Wrap OrbitControls like: 'new OrbitControls(camera, container)'.
- Responsive: The code should adapt if the container resizes (renderer size fits container clientWidth and clientHeight).

Example structure of generated code:
\`\`\`javascript
const width = container.clientWidth;
const height = container.clientHeight;

// 1. Scene, Camera, Renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0a0a0f');
scene.focus = new THREE.FogExp2('#0a0a0f', 0.015);

const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
camera.position.set(0, 10, 20);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(width, height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

// 2. Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// 3. Grid & Helpers
const grid = new THREE.GridHelper(50, 50, '#ff3366', '#222233');
grid.position.y = -1;
scene.add(grid);

// 4. Lights
const ambientLight = new THREE.AmbientLight('#ffffff', 0.1);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight('#4d94ff', 1.5);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// 5. Procedural Texture for Glowing Particles
const canvas = document.createElement('canvas');
canvas.width = 16;
canvas.height = 16;
const ctx = canvas.getContext('2d');
const grad = ctx.createRadialGradient(8, 8, 0, 8, 8, 8);
grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
ctx.fillStyle = grad;
ctx.fillRect(0, 0, 16, 16);
const particleTexture = new THREE.CanvasTexture(canvas);

// Let's keep local mutable param values
let simSpeed = currentParams.speed || 1.0;
let physicsGravity = currentParams.gravity !== undefined ? currentParams.gravity : 9.81;

// ... Create mesh, geometries, etc. ...

// 6. Animation loop
let animationId = null;
const clock = new THREE.Clock();

function animate() {
  animationId = requestAnimationFrame(animate);
  const delta = clock.getDelta() * simSpeed;
  
  // physics updates using delta & physicsGravity
  // ...
  
  controls.update();
  renderer.render(scene, camera);
}
animate();

// 7. Cleanup & Update actions
return {
  updateParams: (newParams) => {
    if (newParams.speed !== undefined) simSpeed = newParams.speed;
    if (newParams.gravity !== undefined) physicsGravity = newParams.gravity;
    // update colors, scales, geometry properties, or lighting here smoothly
  },
  destroy: () => {
    cancelAnimationFrame(animationId);
    controls.dispose();
    renderer.dispose();
    if (renderer.domElement && renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
    // Traverse and dispose materials/geometries
    scene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose());
        } else {
          obj.material.dispose();
        }
      }
    });
    particleTexture.dispose();
  }
};
\`\`\`

You MUST write highly customized Three.js code tailored to the science/math/biology/chemistry concept requested, using strict, isolated rendering templates for each subject:

1. BIOLOGY, CLINICAL MEDICINE, NEUROLOGY, ANATOMY, AND CELULLAR TOPICS (e.g. Neuron, Dendrites, Myelin Axial Nerve, DNA Helix, Cell Structures, Mitochondria, Membranes, Bacterium, Receptors, Hodgkin-Huxley Membrane Channels):
   - Visual Style: ONLY generate organic, biological, and anatomical 3D objects. Use smooth, hand-crafted organic spline/curves, soft translucent lipid membranes (glowing translucent spheres/toruses with low opacity, Fresnel gradients, or double-sided glass properties), and dynamic flowing ion particles representing cellular signals or action potentials.
   - For a Neuron: Draw a beautifully textured branching Soma seed shell with custom organic bezier curves representing dendritic trees, a long myelinated Axon cylinder path, and pulsing neon particle flows representing active neurotransmitters or active electric potentials.
   - For DNA Double Helix: Draw two distinct, synchronized helical curves built from primary glowing molecules connected by horizontal nucleobase rod pairs (A-T, G-C) in alternating vivid colors.
   - For Nerve Cell Membranes and Electrophysiology (e.g., Hodgkin-Huxley simulation): Represent a clean, professional, biochemically correct cell membrane showing custom lipid bilayer rows (two rows of neat lipid head spheres/tails) and embedded ion channels (sodium channel, potassium channel, and leakage channels represented as translucent hollow cylinders). Sodium ions (Na+) must be represented as tiny blue glowing spheres, and Potassium ions (K+) as tiny orange glowing spheres. Animate them actively passing through their specific channel pumps in response to a propagating biological action-potential wave traversing across the axon membrane sphere/grid sequence.
   - CRITICAL RESTRICTION: You are STRICTLY FORBIDDEN from drawing artificial 2D mathematical coordinate axes, oscilloscope waveforms, engineering grids (like black mathematical floor lines), or dry electrical schema components (resistors, diodes, capacitors, batteries, schematic circuit loops) overlaying the biological cells or tissue membranes. Keep the environment purely biology-centric, organic, and clinical-grade clean, representing a microscope/wet-lab chamber.

2. PHYSICS, FLUIDS, ELECTROMAGNETISM, MECHANICS AND OPTICS (e.g., Lorentz Forces, Field Lines, Solar Orbits, Atomic Models, Double Slit diffraction, Wave Interference, Lenses):
   - Visual Style: Focus strictly on mathematical physics models, dynamic field vector arrows, electrostatic flux lines, and exact mechanical/electronic/optical simulations.
   - For Field Lines / Forces: Render neat glowing lines with arrows outlining directional field potential, and point charged particles tracing realistic trajectories guided by Lorentz, gravitational, or Coulomb forces. For an Electric Dipole simulation, you MUST strictly use: tracer particles styled as bright cyan '#00ffd2' (glowing, with emissive properties), positive charge styled as bright red '#ff3366', negative charge styled as bright yellow '#ffff00', and add vibrant PointLights (point light objects) on both positive and negative charge meshes for maximum radiant visibility and exceptional glow.
   - For Optics: Render beautiful glowing laser rays bending dynamically through glass lenses as governed by Snell's Law refraction.
   - For Quantum Mechanics, Orbitals, and Electron Clouds: You must NEVER generate chemical ball-and-stick molecules, CPK atom meshes, or static chemical crystal lattices. Instead, strictly render a high-density probability distribution vertex cloud using 'THREE.Points' mapped with phase-based wave functions (representing Spherical Harmonics lobes for s, p, d, or f orbitals). Color positive phase wave lobes in flowing Cyan/Teal and negative phase wave lobes in Hot Pink/Magenta.

3. CHEMISTRY, MOLECULAR DYNAMICS, AND CRYSTALLOGRAPHY (e.g., Lattice Crystals, Polymers, Chemical Reactions, Catalysis, Molecular Orbitals):
   - Visual Style: Generate ball-and-stick atomic crystals or CPK-colored spheres connected by polished cylinder bonds.
   - Apply classic CPK atomic colors (Carbon: dark grey #334155, Hydrogen: white #f8fafc, Oxygen: red #ef4444, Nitrogen: blue #3b82f6, Sulfur: yellow #eab308, Phosphorous: orange #f97316). Use glowing point-cloud clusters for dynamic electronic orbital shapes.

4. MATHEMATICS, CALCULUS, GEOMETRY, AND CHAOTIC SYSTEMS (e.g., Lorenz Chaotic Attractor, Calabi-Yau, Clifford Attractor, Mobius Strip, Fractals):
   - Visual Style: Abstract pure geometry, high-fidelity manifold surfaces, vector grids, and mathematical curves.
   - For chaotic attractors, render continuous, infinite gradient trail curves using Ribbon/Tube curves or custom Point arrays, tracing precise mathematical differential updates.

Ensure the layout is robust, visual, interactive, and beautiful!

CRITICAL ANTI-BLACK-SCREEN COLOR MANDATE — THIS IS THE HIGHEST PRIORITY RULE:
You are STRICTLY FORBIDDEN from generating ANY invisible or dark-colored elements. Black screen is the worst possible outcome. Follow these rules WITHOUT EXCEPTION:

🚨 BANNED COLORS (will cause black screen — NEVER USE):
- Dark green (#006400, #228B22, #2d5a27, or any dark green)
- Dark purple (#4B0082, #800080, #2d0052, or any dark purple)
- Dark blue (#00008B, #0000CD, #191970, or any dark navy/dark blue)
- Dark red (#8B0000, #800000, or any dark red)
- Dark grey (#333333, #444444, #555555 for main objects)
- ANY color with brightness below 40% on dark background

✅ MANDATORY BRIGHT COLORS (use ONLY these for all 3D objects):
- Bright Cyan: #00ffd2 or #00f3ff
- Electric Lime: #39ff14 or #caff42  
- Hot Pink/Magenta: #ff0077 or #ff55bb
- Vibrant Orange: #ffaa00 or #ff6600
- Bright Yellow: #ffff00 or #ffd700
- Bright Red: #ff3366 or #ff4444
- Bright White: #ffffff or #f0f0f0
- Electric Blue: #0099ff or #00aaff (ONLY this shade, not dark blue)

✅ MANDATORY MATERIAL SETTINGS for ALL objects:
- Always set emissive color matching the object color
- Always set emissiveIntensity between 0.5 and 1.0
- Use MeshStandardMaterial or MeshPhysicalMaterial with emissive
- NEVER use MeshBasicMaterial without bright colors
- For lines: use bright neon colors with linewidth > 1

✅ MANDATORY LIGHTS — Add ALL of these to EVERY scene:
- AmbientLight('#ffffff', 0.4) — ensures base visibility
- DirectionalLight('#ffffff', 1.5) — main light
- At least 2 PointLights with bright colors near main objects

✅ MANDATORY BACKGROUND:
- scene.background = new THREE.Color('#0a0a0f') — pure dark, not black
- Add subtle fog: scene.fog = new THREE.FogExp2('#0a0a0f', 0.01)

SELF-CHECK BEFORE SUBMITTING CODE:
Ask yourself: "If I render this on a dark background, will EVERY element be clearly visible?" 
If ANY element might be invisible → change its color to bright neon immediately!

CRITICAL EXPLANATION & MATHEMETICAL RULES:
- ABSOLUTELY NO PLAIN KEYBOARD-STYLE EQUATIONS ALLOWED. You are strictly FORBIDDEN from outputting formulas in plain keyboard text or plain text (e.g., never use "/" for division, "*" or "·" for multiplication, or keyboard carets "^" or subscripts "_" in plain text). 
- Every single equation, mathematical expression, physical constant, variable, parameter, or chemical reaction across Physics, Chemistry, and Mathematics MUST be perfectly formatted in standard, publication-ready LaTeX.
- Wrap all standalone major/core equations in display math blocks (surrounded by $$) and keep them beautifully centered. Use "\\frac{num}{den}" for all fractions.
- Wrap all individual variables, constants, symbols, small formulas, and inline mathematical values in inline math blocks (surrounded by a single $). For example, inline references must be written as $x$, $y$, $F$, $m_1$, $\\sigma$, etc.
- Always use standard LaTeX symbols: "\\cdot" for multiplication, "\\frac{a}{b}" for division/fractions, and proper subscripts "_" and superscripts "^" within math blocks.
- For Chemical formulas, use standard LaTeX subscripts like $\\text{H}_2\\text{O}$ or $\\text{CO}_2$.
- For Calculus and mathematics, use proper derivatives like $\\frac{dx}{dt} = \\sigma(y - x)$ and elegant integrals.
- Do NOT output raw HTML or plain text math under any circumstances. Every single math or scientific representation must compile perfectly inside standard KaTeX.`;

    const responseSchemaObj = {
      type: Type.OBJECT,
      required: ["topic", "explanation", "threeJsCode", "controlsGuide", "interactiveParameters", "theoryData"],
      properties: {
        topic: {
          type: Type.STRING,
          description: "The name of the science, math, or biology topic.",
        },
        explanation: {
          type: Type.STRING,
          description: "Extremely concise Markdown-formatted explanation containing Scientific Grounding, Step-by-Step Derivation, and 2 sweet, brief practice questions and solutions.",
        },
        threeJsCode: {
          type: Type.STRING,
          description: "The highly optimized, under 120 lines, ready-to-evaluate JavaScript content. Must match systemInstruction exactly.",
        },
        controlsGuide: {
          type: Type.STRING,
          description: "Very brief 1-2 sentences on how to view and interact with the simulation.",
        },
        interactiveParameters: {
          type: Type.ARRAY,
          description: "Sliders or toggles the frontend UI can generate to feed values dynamically into the Three.js updateParams callback.",
          items: {
            type: Type.OBJECT,
            required: ["name", "label", "type", "default"],
            properties: {
              name: { type: Type.STRING, description: "Variable key, e.g. 'gravity' or 'speed'" },
              label: { type: Type.STRING, description: "Display name, e.g. 'Gravity strength'" },
              type: { type: Type.STRING, description: "Input type. Must be 'slider' or 'toggle'." },
              min: { type: Type.NUMBER, description: "Minimum value (required if type is slider)" },
              max: { type: Type.NUMBER, description: "Maximum value (required if type is slider)" },
              step: { type: Type.NUMBER, description: "Step size (required if type is slider)" },
              default: { type: Type.NUMBER, description: "The default value. Number for slider, boolean for toggle." }
            }
          }
        },
        theoryData: {
          type: Type.OBJECT,
          description: "Structured educational content for the Scientific Grounding panel.",
          required: ["conceptAndFormula", "derivation", "realLifeApplications", "practiceQuestions"],
          properties: {
            conceptAndFormula: {
              type: Type.STRING,
              description: "A concise explanation of the core concept and its primary formula(s). All math must be in LaTeX: wrap display equations in $$ and inline math in $.",
            },
            derivation: {
              type: Type.STRING,
              description: "A numbered step-by-step mathematical derivation. Every equation must be in LaTeX (wrapped in $$ or $).",
            },
            realLifeApplications: {
              type: Type.ARRAY,
              description: "Exactly 2 to 3 real-world application strings describing how this concept is used in practice.",
              items: { type: Type.STRING },
            },
            practiceQuestions: {
              type: Type.ARRAY,
              description: "Exactly 2 practice question objects, each with a concise question and a direct answer.",
              items: {
                type: Type.OBJECT,
                required: ["question", "answer"],
                properties: {
                  question: { type: Type.STRING, description: "A brief, direct practice question." },
                  answer: { type: Type.STRING, description: "A concise, precise answer (with LaTeX math where needed)." },
                },
              },
            },
          },
        },
      }
    };

    // Initialize active clients for key rotation
    const rotationClients: { name: string; client: GoogleGenAI }[] = [];
    if (process.env.GEMINI_API_KEY) {
      rotationClients.push({
        name: "GEMINI_API_KEY (Primary)",
        client: new GoogleGenAI({
          apiKey: process.env.GEMINI_API_KEY,
          httpOptions: { headers: { "User-Agent": "aistudio-build" } }
        })
      });
    }
    if (process.env.GEMINI_API_KEY_2) {
      rotationClients.push({
        name: "GEMINI_API_KEY_2 (Secondary)",
        client: new GoogleGenAI({
          apiKey: process.env.GEMINI_API_KEY_2,
          httpOptions: { headers: { "User-Agent": "aistudio-build" } }
        })
      });
    }
    if (process.env.GEMINI_API_KEY_3) {
      rotationClients.push({
        name: "GEMINI_API_KEY_3 (Tertiary)",
        client: new GoogleGenAI({
          apiKey: process.env.GEMINI_API_KEY_3,
          httpOptions: { headers: { "User-Agent": "aistudio-build" } }
        })
      });
    }

    // Helper to generate content with failover rotation across available keys, with optional retries on 503/429
    async function generateContentWithRotation(modelName: string, contents: string, configObj: any) {
      let lastError: any = null;
      for (const { name, client } of rotationClients) {
        let attempts = 0;
        const maxAttempts = 2; // Try up to 2 times for the same key on transient errors
        while (attempts < maxAttempts) {
          try {
            console.log(`[Key Rotation] Sending request for model "${modelName}" using key: ${name} (Attempt ${attempts + 1}/${maxAttempts})...`);
            const res = await client.models.generateContent({
              model: modelName,
              contents,
              config: configObj
            });
            return res;
          } catch (err: any) {
            attempts++;
            const errStr = err?.message || String(err);
            const isTransient = /503|unavailable|overloaded|limit|429/i.test(errStr);
            console.warn(`[Key Rotation] ${name} call failed for model "${modelName}" (Attempt ${attempts}/${maxAttempts}): ${errStr}`);
            lastError = err;
            if (isTransient && attempts < maxAttempts) {
              const backoff = attempts * 600;
              console.log(`[Transient Error] Retrying ${name} for model "${modelName}" in ${backoff}ms...`);
              await new Promise(r => setTimeout(r, backoff));
            } else {
              // Not transient or exhausted attempts, proceed to next key
              break;
            }
          }
        }
      }
      throw lastError || new Error(`All available rotated keys failed to generate content for model "${modelName}"`);
    }



    // ===== FULL PCMB VERIFICATION SYSTEM =====
    function verifyDiagram(queryPrompt: string, generatedData: any): boolean {
      const query = queryPrompt.toLowerCase();
      const code = (generatedData.threeJsCode || "").toLowerCase();
      const explanation = (generatedData.explanation || "").toLowerCase();
      const topic = (generatedData.topic || "").toLowerCase();
      const combined = code + explanation + topic;

      // ---- CHEMISTRY ----
      if (/graphene/i.test(query)) {
        if (/nacl|sodium|chloride/i.test(combined)) { console.warn(`[VERIFY FAIL] Graphene→NaCl mismatch`); return false; }
        if (!/hexagonal|honeycomb|carbon|sp2/i.test(combined)) { console.warn(`[VERIFY FAIL] Graphene missing hexagonal`); return false; }
      }
      if (/\bnacl\b|sodium chloride/i.test(query)) {
        if (!/sodium|chloride|ionic|nacl/i.test(combined)) { console.warn(`[VERIFY FAIL] NaCl missing ionic structure`); return false; }
      }
      if (/\bdna\b|double helix/i.test(query)) {
        if (!/helix|strand|nucleotide|base.*pair|adenine|thymine/i.test(combined)) { console.warn(`[VERIFY FAIL] DNA missing helix`); return false; }
      }
      if (/\bbenzene\b/i.test(query)) {
        if (!/hexagonal|ring|c6h6|aromatic|benzene/i.test(combined)) { console.warn(`[VERIFY FAIL] Benzene missing ring`); return false; }
      }
      if (/\bwater\b|h2o/i.test(query)) {
        if (!/oxygen|hydrogen|h2o|bent|water/i.test(combined)) { console.warn(`[VERIFY FAIL] Water missing molecule`); return false; }
      }
      if (/\bdiamond\b/i.test(query)) {
        if (!/tetrahedral|carbon|diamond|sp3/i.test(combined)) { console.warn(`[VERIFY FAIL] Diamond missing tetrahedral`); return false; }
      }
      if (/\bfullerene\b|c60|buckyball/i.test(query)) {
        if (!/fullerene|c60|buckyball|pentagon|hexagon/i.test(combined)) { console.warn(`[VERIFY FAIL] Fullerene missing structure`); return false; }
      }
      if (/\bglucose\b/i.test(query)) {
        if (!/glucose|sugar|ring|carbon|hydroxyl/i.test(combined)) { console.warn(`[VERIFY FAIL] Glucose missing structure`); return false; }
      }

      // ---- BIOLOGY ----
      if (/\bneuron\b/i.test(query)) {
        if (!/dendrite|axon|soma|cell body/i.test(combined)) { console.warn(`[VERIFY FAIL] Neuron missing components`); return false; }
      }
      if (/mitochondria/i.test(query)) {
        if (!/cristae|membrane|matrix|mitochondr/i.test(combined)) { console.warn(`[VERIFY FAIL] Mitochondria missing structure`); return false; }
      }
      if (/\bcell\b.*division|mitosis/i.test(query)) {
        if (!/chromosome|spindle|mitosis|division|phase/i.test(combined)) { console.warn(`[VERIFY FAIL] Mitosis missing phases`); return false; }
      }
      if (/photosynthesis/i.test(query)) {
        if (!/chlorophyll|light|glucose|co2|photosynthes/i.test(combined)) { console.warn(`[VERIFY FAIL] Photosynthesis missing elements`); return false; }
      }
      if (/\bheart\b/i.test(query)) {
        if (!/ventricle|atrium|valve|heart|chamber/i.test(combined)) { console.warn(`[VERIFY FAIL] Heart missing chambers`); return false; }
      }
      if (/\bprotein\b/i.test(query)) {
        if (!/amino.*acid|protein|helix|folding|peptide/i.test(combined)) { console.warn(`[VERIFY FAIL] Protein missing structure`); return false; }
      }
      if (/ribosome/i.test(query)) {
        if (!/ribosome|rna|mrna|trna|protein.*synthesis/i.test(combined)) { console.warn(`[VERIFY FAIL] Ribosome missing elements`); return false; }
      }

      // ---- PHYSICS ----
      if (/lorenz|lorenz attractor/i.test(query)) {
        if (!/butterfly|attractor|chaos|sigma|lorenz/i.test(combined)) { console.warn(`[VERIFY FAIL] Lorenz missing attractor`); return false; }
      }
      if (/\bgravity\b|gravitational/i.test(query)) {
        if (!/gravity|gravitational|orbit|mass|force/i.test(combined)) { console.warn(`[VERIFY FAIL] Gravity missing physics`); return false; }
      }
      if (/black hole/i.test(query)) {
        if (!/singularity|accretion|event horizon|black hole/i.test(combined)) { console.warn(`[VERIFY FAIL] Black hole missing structure`); return false; }
      }
      if (/\bwave\b.*interference|double slit/i.test(query)) {
        if (!/interference|wave|diffraction|slit/i.test(combined)) { console.warn(`[VERIFY FAIL] Wave interference missing`); return false; }
      }
      if (/projectile/i.test(query)) {
        if (!/projectile|parabola|trajectory|velocity|angle/i.test(combined)) { console.warn(`[VERIFY FAIL] Projectile missing trajectory`); return false; }
      }
      if (/pendulum/i.test(query)) {
        if (!/pendulum|swing|oscillat|bob|period/i.test(combined)) { console.warn(`[VERIFY FAIL] Pendulum missing oscillation`); return false; }
      }
      if (/electric.*field|coulomb/i.test(query)) {
        if (!/electric|field|charge|coulomb|force/i.test(combined)) { console.warn(`[VERIFY FAIL] Electric field missing`); return false; }
      }
      if (/magnetic.*field|electromagnetism/i.test(query)) {
        if (!/magnetic|field|flux|lorentz|electromagnet/i.test(combined)) { console.warn(`[VERIFY FAIL] Magnetic field missing`); return false; }
      }
      if (/quantum/i.test(query)) {
        if (!/quantum|orbital|wave.*function|probability|electron/i.test(combined)) { console.warn(`[VERIFY FAIL] Quantum missing wave function`); return false; }
      }
      if (/\boptics\b|refraction|snell/i.test(query)) {
        if (!/refraction|snell|lens|light|optic/i.test(combined)) { console.warn(`[VERIFY FAIL] Optics missing refraction`); return false; }
      }

      // ---- MATHEMATICS ----
      if (/fourier/i.test(query)) {
        if (!/fourier|wave|frequency|harmonic|sine/i.test(combined)) { console.warn(`[VERIFY FAIL] Fourier missing waves`); return false; }
      }
      if (/\bfractal\b|mandelbrot|julia/i.test(query)) {
        if (!/fractal|mandelbrot|julia|iteration|complex/i.test(combined)) { console.warn(`[VERIFY FAIL] Fractal missing iteration`); return false; }
      }
      if (/mobius/i.test(query)) {
        if (!/mobius|twist|strip|surface/i.test(combined)) { console.warn(`[VERIFY FAIL] Mobius missing strip`); return false; }
      }
      if (/\btorus\b/i.test(query)) {
        if (!/torus|donut|ring|surface/i.test(combined)) { console.warn(`[VERIFY FAIL] Torus missing shape`); return false; }
      }
      if (/vector.*field/i.test(query)) {
        if (!/vector|field|arrow|gradient|curl/i.test(combined)) { console.warn(`[VERIFY FAIL] Vector field missing`); return false; }
      }

      // ---- GENERIC VERIFICATION FOR ALL OTHER TOPICS ----
      // Check 1: Code must be substantial (not empty/black screen)
      if (code.length < 500) {
        console.warn(`[VERIFY FAIL] Code too short (${code.length} chars) — likely empty/black screen!`);
        return false;
      }

      // Check 2: Topic name must appear somewhere in code or explanation
      const queryWords = queryPrompt.toLowerCase().split(" ").filter(w => w.length >= 2);
      const topicMentioned = queryWords.length === 0 || queryWords.some(word => combined.includes(word));
      if (!topicMentioned) {
        console.warn(`[VERIFY FAIL] Topic "${queryPrompt}" not mentioned in generated content!`);
        return false;
      }

      // Check 3: Must have Three.js scene setup
      if (!/scene|camera|renderer|threejs|three\.js/i.test(code)) {
        console.warn(`[VERIFY FAIL] Missing Three.js scene setup!`);
        return false;
      }

      // Check 4: Must have animation loop
      if (!/requestanimationframe|animate\(\)|animationid/i.test(code)) {
        console.warn(`[VERIFY FAIL] Missing animation loop!`);
        return false;
      }

      console.log(`[VERIFY PASSED] ✅ "${queryPrompt}" verified!`);
      return true;
    }

    const geminiSchemaObj = {
      type: Type.OBJECT,
      required: ["topic", "explanation", "controlsGuide", "interactiveParameters", "theoryData", "visualDesignBlueprint"],
      properties: {
        topic: {
          type: Type.STRING,
          description: "The name of the science, math, or biology topic.",
        },
        explanation: {
          type: Type.STRING,
          description: "Extremely concise Markdown-formatted explanation containing Scientific Grounding, Step-by-Step Derivation, and 2 sweet, brief practice questions and solutions.",
        },
        controlsGuide: {
          type: Type.STRING,
          description: "Very brief 1-2 sentences on how to view and interact with the simulation.",
        },
        interactiveParameters: {
          type: Type.ARRAY,
          description: "Sliders or toggles the frontend UI can generate to feed values dynamically into the Three.js updateParams callback.",
          items: {
            type: Type.OBJECT,
            required: ["name", "label", "type", "default"],
            properties: {
              name: { type: Type.STRING, description: "Variable key, e.g. 'gravity' or 'speed'" },
              label: { type: Type.STRING, description: "Display name, e.g. 'Gravity strength'" },
              type: { type: Type.STRING, description: "Input type. Must be 'slider' or 'toggle'." },
              min: { type: Type.NUMBER, description: "Minimum value (required if type is slider)" },
              max: { type: Type.NUMBER, description: "Maximum value (required if type is slider)" },
              step: { type: Type.NUMBER, description: "Step size (required if type is slider)" },
              default: { type: Type.NUMBER, description: "The default value. Number for slider, boolean for toggle." }
            }
          }
        },
        theoryData: {
          type: Type.OBJECT,
          description: "Structured educational content for the Scientific Grounding panel.",
          required: ["conceptAndFormula", "derivation", "realLifeApplications", "practiceQuestions"],
          properties: {
            conceptAndFormula: {
              type: Type.STRING,
              description: "A concise explanation of the core concept and its primary formula(s). All math must be in LaTeX: wrap display equations in $$ and inline math in $.",
            },
            derivation: {
              type: Type.STRING,
              description: "A numbered step-by-step mathematical derivation. Every equation must be in LaTeX (wrapped in $$ or $).",
            },
            realLifeApplications: {
              type: Type.ARRAY,
              description: "Exactly 2 to 3 real-world application strings describing how this concept is used in practice.",
              items: { type: Type.STRING },
            },
            practiceQuestions: {
              type: Type.ARRAY,
              description: "Exactly 2 practice question objects, each with a concise question and a direct answer.",
              items: {
                type: Type.OBJECT,
                required: ["question", "answer"],
                properties: {
                  question: { type: Type.STRING, description: "A brief, direct practice question." },
                  answer: { type: Type.STRING, description: "A concise, precise answer (with LaTeX math where needed)." },
                },
              },
            },
          },
        },
        visualDesignBlueprint: {
          type: Type.STRING,
          description: "Detailed step-by-step blueprint of the 3D scene structure, lighting, animations, and mathematical updates to pass to the coder.",
        }
      }
    };

    const geminiSystemPrompt = `You are an elite scientific advisor, physicist, mathematician, chemist, and biologist.
Your task is to take the user's exact science, mathematical, chemical, biological, or medical topic query: "${prompt}".
You MUST explain this query with high precision and design a gorgeous, interactive 3D visual simulation blueprint for it.

⚠️ CONCISE MULTI-SECTION GENERATION DIRECTIVE:
1. In the "explanation" field, generate a concise Markdown text containing exactly these three sub-sections:
   - ### Scientific Grounding: Short bullet points and key formulas (using standard LaTeX formulas wrapped in $$ or $).
   - ### Step-by-Step Derivation: Short bullet points and key formulas (using standard LaTeX formulas wrapped in $$ or $).
   - ### Practice Questions: Exactly 2 sweet, brief, and direct practice questions with their solutions.
2. Keep all of these explanation sections very short, sweet, and direct to prevent model timeouts or exceeding token limits.
3. Keep "controlsGuide" very brief (1-2 sentences on how to interact).
4. You MUST populate the "theoryData" object with ALL four of the following fields — no field may be omitted or left empty:
   - "conceptAndFormula": A concise explanation of the core concept and its primary formula(s). Wrap all math in $$ for display blocks or $ for inline. Use LaTeX for every equation.
   - "derivation": A numbered step-by-step mathematical derivation of the concept. Use LaTeX for every equation (wrapped in $$ or $).
   - "realLifeApplications": Exactly 2 to 3 real-world application strings (plain text sentences) describing how this concept is used in practice.
   - "practiceQuestions": Exactly 2 objects each with a "question" string and an "answer" string. Keep both concise and direct.
5. In the "visualDesignBlueprint" field, write a comprehensive, highly detailed description of how to render this concept in Three.js. Specify:
   - What geometry shapes, meshes, lines, particles, and custom neon colors should be created.
   - What lights to place and where (ambient, directional, point lights).
   - The exact mathematical and physical equations or differential integrations (like Verlet, Euler, or attractor coordinate updates) to calculate inside the animation loop.
   - How each of the sliders/toggles in "interactiveParameters" should dynamically affect the simulation variables in the updateParams(newParams) callback.

${structuralHint ? `⚠️ CRITICAL STRUCTURE OVERRIDE — READ THIS FIRST:\n${structuralHint}\n` : ""}`;

    let finalData: any = null;
    let feedback: string = "";
    let success = false;

    // Unified Robust Self-Correction & Model Fallback Retry Loop (Up to 3 Attempts)
    for (let attempt = 1; attempt <= 3; attempt++) {
      console.log(`[Generation Pipeline] Attempt ${attempt}/3 starting for concept: "${prompt}"`);

      if (attempt > 1) {
        const attemptDelay = (attempt - 1) * 1000;
        console.log(`[Generation Pipeline] Adding inter-attempt backoff delay of ${attemptDelay}ms to clear server spikes...`);
        await new Promise(r => setTimeout(r, attemptDelay));
      }

      // Determine model to use with extreme resilience across different model families
      let modelToUse = "gemini-3.5-flash";
      if (thinkingMode === true) {
        if (attempt === 1) {
          modelToUse = "gemini-3.1-pro-preview";
        } else if (attempt === 2) {
          modelToUse = "gemini-3.5-flash";
        } else {
          modelToUse = "gemini-3.1-flash-lite";
        }
      } else {
        if (attempt === 1) {
          modelToUse = "gemini-3.5-flash";
        } else if (attempt === 2) {
          modelToUse = "gemini-3.1-flash-lite";
        } else {
          modelToUse = "gemini-3.5-flash";
        }
      }

      try {
        let parsedData: any;

        if ((modelProvider === "zai" || modelProvider === "hybrid") && hasOpenRouterKey) {
          // --- HYBRID PIPELINE (Gemini Science Logic + OpenRouter Three.js Coder) ---
          console.log(`[Hybrid Pipeline] Phase 1: Calling Gemini "${modelToUse}" for scientific schema & blueprint...`);
          
          let currentPromptContent = `Formulate a precise scientific, math, or biological explanation and a 3D visual design blueprint specifically for the topic: "${prompt}".`;
          if (feedback) {
            currentPromptContent += `\n\n⚠️ CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT:\n${feedback}\nPlease correct any errors and adjust the visualization/scientific logic accordingly.`;
          }

          const geminiResponse = await generateContentWithRotation(modelToUse, currentPromptContent, {
            systemInstruction: geminiSystemPrompt,
            responseMimeType: "application/json",
            responseSchema: geminiSchemaObj
          });

          const geminiText = geminiResponse.text;
          if (!geminiText || geminiText.trim() === "") {
            throw new Error("Gemini returned empty scientific logic");
          }

          const geminiResult = JSON.parse(geminiText.trim());

          console.log(`[Hybrid Pipeline] Phase 2: Requesting Three.js code from OpenRouter free model cohere/north-mini-code...`);
          const openRouterCode = await callOpenRouterCodeGenerator(
            geminiResult.topic || prompt,
            geminiResult.visualDesignBlueprint,
            geminiResult.interactiveParameters || []
          );

          parsedData = {
            topic: geminiResult.topic,
            explanation: geminiResult.explanation,
            controlsGuide: geminiResult.controlsGuide,
            interactiveParameters: geminiResult.interactiveParameters,
            theoryData: geminiResult.theoryData,
            threeJsCode: openRouterCode
          };
        } else {
          // --- FALLBACK (Single Gemini call generating everything directly) ---
          console.log(`[Single Pipeline Fallback] Calling Gemini "${modelToUse}" to generate everything in one step...`);
          
          let currentPromptContent = `Explain and generate high-fidelity, completely custom interactive 3D rendering code specifically for the query: "${prompt}". 
Ensure that all the shapes, parameters, labels, formulas, and animations directly and strictly visualize "${prompt}". Do not generalize to another popular topic or fallback to standard templates.`;
          if (feedback) {
            currentPromptContent += `\n\n⚠️ CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT:\n${feedback}\nPlease pay extremely close attention to the requested topic and solve these errors.`;
          }

          const geminiResponse = await generateContentWithRotation(modelToUse, currentPromptContent, {
            systemInstruction: systemPrompt,
            responseMimeType: "application/json",
            responseSchema: responseSchemaObj
          });

          const geminiText = geminiResponse.text;
          if (!geminiText || geminiText.trim() === "") {
            throw new Error("Gemini returned empty content");
          }

          parsedData = JSON.parse(geminiText.trim());
        }

        // Verify Three.js Code and Topic Specificity
        const isVerified = verifyDiagram(prompt, parsedData);
        if (isVerified) {
          finalData = parsedData;
          success = true;
          console.log(`[Generation Pipeline] ✅ Successfully generated and verified simulation for "${prompt}" on attempt ${attempt}`);
          break; // Exit retry loop early!
        } else {
          console.warn(`[Verification Fail] Attempt ${attempt} failed the verification guidelines.`);
          feedback = `The simulation you generated for "${prompt}" failed our verification rules. Please:
1. Ensure the generated Three.js code contains a complete scene setup, bright neon materials, directional lighting, OrbitControls, and a requestAnimationFrame loop.
2. The generated shapes, formulas, labels, and parameters MUST strictly visualize "${prompt}". Do not return a generic fallback or NaCl structure.
3. Keep the code lightweight, robust, and clean. Make sure the background is black and the visualization uses bright emissive/glowing components.`;
        }
      } catch (err: any) {
        console.warn(`[Generation Pipeline] Attempt ${attempt} model call failed with error:`, err?.message || String(err));
        feedback = `API or Generation Error occurred: ${err?.message || String(err)}. Please try another generation approach.`;
      }
    }

    if (success && finalData) {
      generationCache.set(cachedKey, finalData);
      return res.json(finalData);
    } else {
      console.warn(`[Generation Pipeline] All 3 attempts failed to generate a verified simulation for "${prompt}".`);

      if (hasManualFallback(prompt)) {
        console.log(`[Fallback Triggered] Manual fallback template found for "${prompt}". Serving local high-fidelity template.`);
        const fallbackVal = getFallbackResponse(prompt, "Model generation failed. Loaded high-fidelity manual fallback template.");
        generationCache.set(cachedKey, fallbackVal);
        return res.json(fallbackVal);
      } else {
        console.log(`[Pipeline Error] No manual fallback template for "${prompt}". Returning honest error message.`);
        return res.status(400).json({
          error: "Is topic ke liye abhi simulation generate nahi ho saka, dusra topic try karo ya thodi der baad try karo"
        });
      }
    }
  } catch (error: any) {
    console.warn(`[Pipeline Error Exception] Caught exception during generation:`, error?.message || String(error));
    if (hasManualFallback(prompt)) {
      const fallbackVal = getFallbackResponse(prompt, error.message || String(error));
      generationCache.set(cachedKey, fallbackVal);
      return res.json(fallbackVal);
    } else {
      return res.status(400).json({
        error: "Is topic ke liye abhi simulation generate nahi ho saka, dusra topic try karo ya thodi der baad try karo"
      });
    }
  }
});

// Endpoint to delete/flush all cached generation responses
app.post("/api/clear-cache", (req, res) => {
  generationCache.clear();
  console.log("[Cache Cleared] Internal memory generation cache successfully purged.");
  res.json({ success: true, message: "Server-side simulation caches fully cleared." });
});

// Serve frontend assets in development and production
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

const isNetlify = process.env.NETLIFY === "true" || process.env.LAMBDA_TASK_ROOT;

if (!isNetlify) {
  startServer();
}

export { app };
