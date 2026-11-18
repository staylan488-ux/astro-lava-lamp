const fs = require('fs');
const path = require('path');

const dir = __dirname;
let html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(dir, 'style.css'), 'utf8');

html = html.replace(/<link[^>]*href="style\.css[^"]*"[^>]*>/, () => `<style>\n${css}\n</style>`);

for (const file of ['physics.js', 'shaders.js', 'renderer.js', 'main.js']) {
  const js = fs.readFileSync(path.join(dir, file), 'utf8');
  const escaped = file.replace('.', '\\.');
  const re = new RegExp(`<script[^>]*src="${escaped}[^"]*"[^>]*></script>`);
  html = html.replace(re, () => `<script>\n${js}\n</script>`);
}

const outPath = path.join(dir, 'lava-lamp-standalone.html');
fs.writeFileSync(outPath, html, 'utf8');

const sizeKb = (fs.statSync(outPath).size / 1024).toFixed(1);
const extRefs = (html.match(/\b(src|href)=/g) || []).length;
console.log(`Bundled ${outPath} (${sizeKb} KB, external refs: ${extRefs})`);
