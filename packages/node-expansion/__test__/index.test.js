const { getDescendants } = require('../src/index');

//CHEBI:26218 -> CHEBI:50394 -> CHEBI:51700
//potassium salt -> organic potassium salt -> apalcillin potassium
test('chebi', () => {
  let curies = ['CHEBI:26218'];
  let descendants = getDescendants(curies);
  expect(descendants['CHEBI:26218']).toContain('CHEBI:50394');
  expect(descendants['CHEBI:26218']).toContain('CHEBI:51700');
});

//DOID:0060524 -> DOID:0040001 -> DOID:0060527
//crustacean allergy -> shrimp allergy -> indian prawn allergy
test('doid', () => {
  let curies = ['DOID:0060524'];
  let descendants = getDescendants(curies);
  expect(descendants['DOID:0060524']).toContain('DOID:0040001');
  expect(descendants['DOID:0060524']).toContain('DOID:0060527');
});

//GO:0032553 -> GO:0032555 -> GO:0032559 
//ribonucleotide binding -> purine ribonucleotide binding -> adenyl ribonucleotide binding
test('go', () => {
  let curies = ['GO:0032553'];
  let descendants = getDescendants(curies);
  expect(descendants['GO:0032553']).toContain('GO:0032555');
  expect(descendants['GO:0032553']).toContain('GO:0032559');
});

//HP:0003812 -> HP:0003829 -> HP:0003831
//Phenotypic variability -> Incomplete penetrance -> Age-dependent penetrance
test('hp', () => {
  let curies = ['HP:0003812'];
  let descendants = getDescendants(curies);
  expect(descendants['HP:0003812']).toContain('HP:0003829');
  expect(descendants['HP:0003812']).toContain('HP:0003831');
});

//MONDO:0001517 -> MONDO:0001955 -> MONDO:0024275
//dysentery -> protozoal dysentery -> amebic dysentery
test('mondo', () => {
  let curies = ['MONDO:0001517'];
  let descendants = getDescendants(curies);
  expect(descendants['MONDO:0001517']).toContain('MONDO:0001955');
  expect(descendants['MONDO:0001517']).toContain('MONDO:0024275');
});

//UMLS:C0680344 -> UMLS:C0003114 -> UMLS:C0680369
//sociocultural values, norms, and social control -> anomie -> anomie(normless)
test('umls', () => {
  let curies = ['UMLS:C0680344'];
  let descendants = getDescendants(curies);
  expect(descendants['UMLS:C0680344']).toContain('UMLS:C0003114');
  expect(descendants['UMLS:C0680344']).toContain('UMLS:C0680369');
});

test('multiple', () => {
  let curies = ['HP:0003812', 'MONDO:0001517'];
  let descendants = getDescendants(curies);
  expect(descendants['HP:0003812']).toContain('HP:0003829');
  expect(descendants['HP:0003812']).toContain('HP:0003831');
  expect(descendants['MONDO:0001517']).toContain('MONDO:0001955');
  expect(descendants['MONDO:0001517']).toContain('MONDO:0024275');
});

test('no descendants', () => {
  let curies = ['DOID:0060527'];
  let descendants = getDescendants(curies);
  expect(descendants['DOID:0060527'].length).toBe(0);
});

test('recursive', () => {
  let curies = ['UMLS:C0012634'];
  let descendants = getDescendants(curies);
});