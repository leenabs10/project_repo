/* Exercise generated editor state and source-bound RLE exports without browser I/O. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

async function main() {
  const html = fs.readFileSync(process.argv[2], 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const width = 4, height = 2;
  const candidate = [0, 1, 1, 0, 0, 0, 1, 0];
  const pixels = new Uint8ClampedArray(width * height * 4);
  candidate.forEach((value, index) => { pixels[index * 4] = value; pixels[index * 4 + 3] = 255; });
  function canvas() {
    return {getContext: () => ({clearRect() {}, putImageData() {}, drawImage() {},
      createImageData: () => ({data: new Uint8ClampedArray(width * height * 4)}),
      getImageData: () => ({data: pixels})}),
      getBoundingClientRect: () => ({left: 0, top: 0, width, height}), setPointerCapture() {}};
  }
  const elements = {'#mask': canvas(), '#rgb': canvas(), '#class': {value: '255'}, '#radius': {value: '1'}, '#confirmed': {}, '#reviewer': {}};
  const document = {
    querySelector: name => elements[name] ||= {},
    createElement: name => name === 'canvas' ? canvas() : {click() {}},
  };
  let exported = null;
  const sandbox = {document, Uint8Array, Math, Date, JSON, Promise,
    Image: class { constructor() { this.naturalWidth = width; this.naturalHeight = height; } set src(value) { queueMicrotask(() => this.onload()); } },
    Blob: class { constructor(parts) { exported = parts.join(''); } },
    URL: {createObjectURL: () => 'fixture-url', revokeObjectURL() {}},
    setTimeout: callback => callback(), alert: message => { throw Error(message); },
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(vm.runInContext('ready', context), true);
  assert.deepEqual(Array.from(vm.runInContext('labels', context)), candidate);
  elements['#confirmed'].checked = true;
  elements['#mask'].onpointerdown({clientX: 1, clientY: 1, pointerId: 1, preventDefault() {}});
  elements['#mask'].onpointerup();
  assert.equal(elements['#confirmed'].checked, false);
  assert.equal(vm.runInContext('labels[5]', context), 255);
  elements['#undo'].onclick();
  assert.deepEqual(Array.from(vm.runInContext('labels', context)), candidate);
  elements['#reviewer'].value = 'Fixture editor operator';
  elements['#confirmed'].checked = true;
  elements['#export'].onclick();
  const submission = JSON.parse(exported);
  assert.equal(submission.reviewed, true);
  assert.equal(submission.width, width);
  assert.equal(submission.height, height);
  assert.deepEqual(submission.mask_rle.flatMap(([value, count]) => Array(count).fill(value)), candidate);
  process.stdout.write(JSON.stringify(submission));
}
main().catch(error => { process.stderr.write(String(error.stack)); process.exitCode = 1; });
