/* Known 4x2 labels test editor behavior, never attest real customer images. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
/** Simulate editor controls with known labels and verify exported RLE without real image approval. */
async function main() {
  const html = fs.readFileSync(process.argv[2], 'utf8');
  const candidate = JSON.parse(process.argv[3]);
  const width = 4, height = 2;
  const sourcePixels = new Uint8ClampedArray(width * height * 4);
  candidate.forEach((value, index) => {
    sourcePixels[index * 4] = value;
    sourcePixels[index * 4 + 3] = 255;
  });
  let displayed = null, exported = null;
  const elements = {
    '#surface': {
      style: {
      }
    },
    '#view': {
      value: 'overlay'
    },
    '#zoom': {
      value: 'fit'
    },
    '#radius': {
      value: '1'
    },
    '#class': {
      value: '0'
    },
    '#reviewer': {
      value: ''
    },
    '#confirmed': {
      checked: false
    }
  };
  function canvas() {
    return {
      getContext: () => ( {
        clearRect() {
        },
        drawImage() {
        },
        createImageData: () => ( {
          data: new Uint8ClampedArray(width * height * 4)
        }),
        getImageData: () => ( {
          data: sourcePixels
        }),
        putImageData(image) {
          displayed = image.data.slice();
        }
      }),
      getBoundingClientRect: () => ( {
        left: 10,
        top: 20,
        width: parseInt(elements['#surface'].style.width) || 4,
        height: (parseInt(elements['#surface'].style.width) || 4) / 2
      }),
      setPointerCapture() {
      }
    };
  }
  elements['#mask'] = canvas();
  elements['#rgb'] = canvas();
  const document = {
    querySelector: key => elements[key] ||= {
    },
    createElement: name => name === 'canvas' ? canvas(): {
      click() {
      }
    }
  };
  const sandbox = {
    document,
    Uint8Array,
    Math,
    Date,
    JSON,
    Promise,
    Error,
    fetch: async() => ( {
      ok: false
    }),
    Image: class {
      constructor() {
        this.naturalWidth = width;
        this.naturalHeight = height;
      }
      set src(value) {
        queueMicrotask(() => this.onload());
      }
    },
    Blob: class {
      constructor(parts) {
        exported = parts.join('');
      }
    },
    URL: {
      createObjectURL: () => 'fixture-url',
      revokeObjectURL() {
      }
    },
    setTimeout: callback => callback()
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(vm.runInContext('ready', context), true);
  assert.deepEqual(Array.from(vm.runInContext('labels', context)), candidate);
  assert.equal(elements['#confirmed'].checked, false);
  for (let index = 0; index < candidate.length; index++) {
    const expected = candidate[index] === 1 ? [235, 70, 210]: candidate[index] === 0 ? [0, 180, 210]: [255, 160, 0];
    assert.deepEqual(Array.from(displayed.slice(index * 4, index * 4 + 3)), expected);
  }
  elements['#export'].onclick();
  assert.equal(exported, null);
  elements['#zoom'].value = '2';
  elements['#zoom'].onchange();
  assert.equal(elements['#surface'].style.width, '8px');
  elements['#confirmed'].checked = true;
  elements['#mask'].onpointerdown( {
    clientX: 14,
    clientY: 22,
    pointerId: 1,
    preventDefault() {
    }
  });
  elements['#mask'].onpointerup();
  assert.equal(vm.runInContext('labels[6]', context), 0);
  assert.equal(elements['#confirmed'].checked, false);
  elements['#undo'].onclick();
  assert.deepEqual(Array.from(vm.runInContext('labels', context)), candidate);
  elements['#view'].value = 'rgb';
  elements['#view'].onchange();
  assert.equal(displayed.every(value => value === 0), true);
  assert.deepEqual(Array.from(vm.runInContext('labels', context)), candidate);
  elements['#view'].value = 'labels';
  elements['#view'].onchange();
  assert.equal(displayed[3], 255);
  elements['#reviewer'].value = 'Known fixture reviewer';
  elements['#confirmed'].checked = true;
  elements['#reviewer'].oninput();
  assert.equal(elements['#confirmed'].checked, false);
  elements['#confirmed'].checked = true;
  elements['#export'].onclick();
  const submission = JSON.parse(exported);
  assert.equal(submission.reviewer_kind, 'human');
  assert.deepEqual(submission.mask_rle.flatMap(([value, count]) => Array(count).fill(value)), candidate);
  process.stdout.write(JSON.stringify(submission));
}
main().catch (error => {
  process.stderr.write(String(error.stack));
  process.exitCode = 1;
});
