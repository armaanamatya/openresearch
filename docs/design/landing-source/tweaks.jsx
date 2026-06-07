/* OpenResearch landing — Tweaks panel */
const { useEffect } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accentHex":   "#E8A24A",
  "bgDepth":     0.16,
  "density":     1,
  "numStyle":    "section",
  "grainOn":     true
}/*EDITMODE-END*/;

// section · paragraph · bracket
const NUM_PRESETS = {
  section:   { glyph: "§", sep: " ", weight: 500 },
  paragraph: { glyph: "¶", sep: " ", weight: 500 },
  bracket:   { glyph: "[", sep: "]", weight: 500 },
  plain:     { glyph: "",  sep: " ", weight: 500 }
};

function hexToOklchVar(hex) {
  // crude: convert hex to oklch via canvas, then return css var string
  // We'll just hand a hex to CSS using a tinted oklch chain. Simpler: set --accent directly to hex.
  return hex;
}

function Tweaks() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty('--accent', t.accentHex);
    root.style.setProperty('--bg',    `oklch(${t.bgDepth} 0.004 270)`);
    root.style.setProperty('--bg-2',  `oklch(${t.bgDepth + 0.025} 0.004 270)`);
    root.style.setProperty('--bg-elev',`oklch(${t.bgDepth + 0.045} 0.004 270)`);

    // density
    const pad = ({
      tight:   '60px',
      normal:  'clamp(72px, 10vw, 160px)',
      roomy:   'clamp(96px, 13vw, 200px)'
    })[['tight','normal','roomy'][t.density]] || 'clamp(72px, 10vw, 160px)';
    root.style.setProperty('--section-pad', pad);

    // numbering style — rewrite §
    const preset = NUM_PRESETS[t.numStyle] || NUM_PRESETS.section;
    document.querySelectorAll('.spec-num .glyph').forEach(el => {
      el.textContent = preset.glyph;
    });
    // also rewrite glyphs in eyebrow tags etc.
    document.querySelectorAll('[data-numglyph]').forEach(el => {
      el.textContent = preset.glyph;
    });

    // grain
    document.body.style.setProperty('--grain-display', t.grainOn ? 'block' : 'none');
    let style = document.getElementById('__grain-style');
    if (!style) {
      style = document.createElement('style');
      style.id = '__grain-style';
      document.head.appendChild(style);
    }
    style.textContent = `body::before { display: ${t.grainOn ? 'block' : 'none'}; }`;
  }, [t]);

  return (
    <TweaksPanel title="Tweaks">
      <TweakSection label="Accent">
        <TweakColor
          label="Accent color"
          value={t.accentHex}
          options={["#E8A24A", "#C0F76A", "#7DD3FC", "#F472B6", "#A78BFA", "#F87171"]}
          onChange={v => setTweak('accentHex', v)}
        />
      </TweakSection>

      <TweakSection label="Surface">
        <TweakSlider
          label="Background depth"
          min={0.08} max={0.22} step={0.01}
          value={t.bgDepth}
          onChange={v => setTweak('bgDepth', v)}
        />
        <TweakToggle
          label="Paper grain"
          value={t.grainOn}
          onChange={v => setTweak('grainOn', v)}
        />
      </TweakSection>

      <TweakSection label="Rhythm">
        <TweakRadio
          label="Density"
          value={t.density}
          options={[
            { value: 0, label: "Tight" },
            { value: 1, label: "Normal" },
            { value: 2, label: "Roomy" }
          ]}
          onChange={v => setTweak('density', v)}
        />
      </TweakSection>

      <TweakSection label="Numbering">
        <TweakRadio
          label="Glyph"
          value={t.numStyle}
          options={[
            { value: "section",   label: "§" },
            { value: "paragraph", label: "¶" },
            { value: "bracket",   label: "[ ]" },
            { value: "plain",     label: "—" }
          ]}
          onChange={v => setTweak('numStyle', v)}
        />
      </TweakSection>
    </TweaksPanel>
  );
}

const root = ReactDOM.createRoot(document.getElementById('tweaks-root'));
root.render(<Tweaks />);
