var fs = require("fs");
const { CHILD_RELATIONS, PARENT_RELATIONS } = require("./config");

const urltoCurie = url => {
  if (url.startsWith("http://purl.obolibrary.org/obo/")) {
    return url.replace("http://purl.obolibrary.org/obo/", "").replace("_", ":");
  }
};

const parseFile = filename => {
  let f = JSON.parse(fs.readFileSync(filename, "utf8"));

  let map_edges = {};

  for (let edge of f.graphs[0].edges) {
    if (CHILD_RELATIONS[edge.pred]) {
      let obj = urltoCurie(edge.obj);
      let sub = urltoCurie(edge.sub);
      if (!map_edges[obj]) {
        map_edges[obj] = [];
      }
      map_edges[obj].push(sub);
    }
    // else if (PARENT_RELATIONS[edge.pred]) {
    //   let obj = urltoCurie(edge.sub);
    //   let sub = urltoCurie(edge.obj);
    //   if (!map_edges[obj])
    //   {
    //     map_edges[obj] = [];
    //   }
    //   map_edges[obj].push(sub);
    // }
  }

  //export map_edges into json file
  fs.writeFileSync(filename.replace(".json", "-parsed.json"), JSON.stringify(map_edges));
};

exports.parseFile = parseFile;
