const { ONTOLOGIES } = require("./config");
const _ = require("lodash");

let data = {};
let loaded = false;

const ENTITY_CAP = 100;

//load data from json files based on ontologies
const loadData = () => {
  if (loaded) return;
  // const start = performance.now();
  for (let ontology in ONTOLOGIES) {
    const filename = `../data/${ontology}-parsed.json`;
    const ontologyData = require(filename);
    data = { ...data, ...ontologyData };
  }
  // const end = performance.now();
  // console.log(`loadData took ${end - start} milliseconds.`);
  loaded = true;
};

//get all children of a list of curies
exports.getDescendants = (curies, recursive = true) => {
  loadData();
  if (recursive) {
    //run level order traversal to get closest descendants
    const children = {};

    for (let curie of curies) {
      children[curie] = [];
      level = [curie];
      while (level.length > 0 && children[curie].length < ENTITY_CAP) {
        next_level = [];
        for (let c of level) {
          if (data[c]) {
            children[curie].push(...data[c]);
            next_level.push(...data[c]);
          }
        }
        level = next_level;
      }
      children[curie] = _.uniq(children[curie])
        .filter(child => child !== curie)
        .slice(0, ENTITY_CAP);
    }
    return children;
  } else {
    return Object.fromEntries(
      Object.entries(_.pick(data, curies)).map(([curie, descendants]) => {
        return [curie, descendants.filter(descendant => descendant !== curie)];
      }),
    );
  }
};
