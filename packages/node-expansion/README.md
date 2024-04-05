# Node Expansion

Module for expanding terms to children.

## Install

```
pnpm i @biothings-explorer/node-expansion
```

## Usage

- by default, the module will expand terms to all descendants

```js
const { getDescendants } = require('@biothings-explorer/node-expansion');

console.log(getDescendants(['GO:0022010', 'DOID:0060524']));

// {
//   'DOID:0060524': [
//     'DOID:0040001',
//     'DOID:0060526',
//     'DOID:0060525',
//     'DOID:0060527',
//     'DOID:0060528',
//     'DOID:0060529'
//   ]
// }
```

 - use `recursive=false` to get direct children only
```js
const { getDescendants } = require('@biothings-explorer/node-expansion');

console.log(getDescendants(['GO:0022010', 'DOID:0060524', 'DOID:4'], recursive=false));

// {
//   'DOID:0060524': [ 'DOID:0040001', 'DOID:0060526' ],
//   'DOID:4': [
//     'DOID:0014667',
//     'DOID:0050117',
//     'DOID:0080015',
//     'DOID:14566',
//     'DOID:150',
//     'DOID:225',
//     'DOID:630',
//     'DOID:7'
//   ]
// }
```

### Currently included ontologies
- GO
- DOID
- HP
- MONDO
- CHEBI

### Update Data Files
```
node download.js
```
 - downloads, converts, and parses all the owl files listed in the config file
