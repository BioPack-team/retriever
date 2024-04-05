# BioLink Model

A nodejs library for manipulating BioLink Model.

[![Test Codecov](https://github.com/biothings/biolink-model.js/actions/workflows/test_cov.yml/badge.svg)](https://github.com/biothings/biolink-model.js/actions/workflows/test_cov.yml)
[![codecov](https://codecov.io/github/biothings/biolink-model.js/branch/main/graph/badge.svg?token=96ZMR2523K)](https://codecov.io/github/biothings/biolink-model.js)

## 💾 Install

```bash
pnpm i biolink-model
```

## 📐 Usage

### 📢 Import and Initialize

```javascript
const bl = require("biolink-model")

const biolink = new bl.BioLink();

```

<br>

### ❗ Load BioLink Yaml File

Loading BioLink Yaml file is a required step before you can utilizing this package to traverse the hierarchy tree for BioLink predicates and classes.

The Yaml file can be loaded through a url, a local file stored within the package, or a valid file path provided.

#### 🔎 Load the BioLink-model yaml file stored along with the package

This can be done in sync or async mode.

```javascript
// load in async mode
await biolink.load();
// in sync mode
biolink.loadSync();
```

#### 🔎 Load the BioLink-model yaml file from a valid url

```javascript
// load in async mode
/// the url provided below points to the most recent version of the biolink model.
await biolink.load("https://raw.githubusercontent.com/biolink/biolink-model/master/biolink-model.yaml");
```

#### 🔎 Load the BioLink-model yaml file from a local file path

This can be done in sync or async mode as well.

```javascript
import path from 'path';
// load in async mode
// assume your biolink file is stored in the same folder under biolink.yaml
await biolink.load(path.resolve(__dirname, './biolink.yaml');
// in sync mode
biolink.loadSync(path.resolve(__dirname, './biolink.yaml');
```

<br>

### ❇️ Traverse the BioLink Class Hierarchy

#### 🔎 Get BioLink Class Tree Object

```javascript
const tree = biolink.classTree;
```

#### 🔎 Get the ancestors of a biolink class

```javascript

const tree = biolink.classTree;

const ancestors = tree.getAncestors("Gene")
```

#### 🔎 Get the descendants of a biolink class

```javascript

const tree = biolink.classTree;

const descendants = tree.getDescendants("MolecularEntity")
```

<br>

### ❇️ Traverse the BioLink Slot Hierarchy

#### 🔎 Get BioLink Slot Tree Object

```javascript
const tree = biolink.slotTree;
```

#### 🔎 Get the ancestors of a biolink slot

```javascript

const tree = biolink.slotTree;

const ancestors = tree.getAncestors("regulates")
```

#### 🔎 Get the descendants of a biolink slot

```javascript

const tree = biolink.slotTree;

const descendants = tree.getDescendants("regulates")
```
